import os
import pickle
from sys import argv

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline, FeatureUnion

import BucketFactory
import ClassifierFactory
import EncoderFactory
from DatasetManager import DatasetManager

train_file = argv[1]
bucket_encoding = "agg"
bucket_method = argv[2]
cls_encoding = argv[3]
cls_method = argv[4]
n_min_cases_in_bucket = int(argv[5])
n_iter = int(argv[6])

dataset_ref = os.path.splitext(train_file)[0]
home_dirs = os.environ['PYTHONPATH'].split(":")
home_dir = home_dirs[0] # if there are multiple PYTHONPATHs, choose the first
logs_dir = "logdata/"
results_dir = "results/CV/"

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}

method_name = "%s_%s" % (bucket_method, cls_encoding)
methods = encoding_dict[cls_encoding]

outfile = os.path.join(home_dir, results_dir,
                       "CV_%s_%s_%s_%s_FA.csv" % (dataset_ref, method_name, cls_method, n_min_cases_in_bucket))

random_state = 22
fillna = True
cls_params_names = ['n_estimators', 'learning_rate', 'subsample', 'max_depth', 'colsample_bytree', 'min_child_weight']

##### MAIN PART ######

with open(outfile, 'w') as fout:
    fout.write("%s;%s;%s;%s;%s;%s;%s\n" % ("label_col", "method", "cls", ";".join(cls_params_names), "nr_events", "metric", "score"))

    dataset_manager = DatasetManager(dataset_ref)
    dtypes = {col: "str" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
              [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
    for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
        dtypes[col] = "float"

    # if dataset_manager.mode == "regr":
    #     dtypes[dataset_manager.label_col] = "float"  # if regression, target value is float
    # else:
    #     dtypes[dataset_manager.label_col] = "str"  # if classification, preserve and do not interpret dtype of label

    data = pd.read_csv(os.path.join(home_dir, logs_dir, train_file), sep=";", dtype=dtypes)
    #data = data.head(30000)
    data[dataset_manager.timestamp_col] = pd.to_datetime(data[dataset_manager.timestamp_col])

    # split data into training and validation sets
    train_, _ = dataset_manager.split_data(data, train_ratio=0.660)
    train_, test_ = dataset_manager.split_data(train_, train_ratio=0.8)

    # consider prefix lengths until 90th percentile of case length
    min_prefix_length = 2
    max_prefix_length = dataset_manager.max_prefix_length
    #max_prefix_length = min(6, dataset_manager.get_pos_case_length_quantile(data, 0.95))
    del data

    # file with activities and gateways to predict
    target_df_ = pd.read_csv(os.path.join(home_dir, logs_dir, "target/target_%s" % train_file),
                            dtype={'%s' % dataset_manager.case_id_col: str})

    gateway_exits = {}
    for gateway in dataset_manager.label_cat_cols:
        gateway_exits[gateway] = target_df_.loc[target_df_[gateway] != -1][gateway].nunique()

    for label_col in dataset_manager.label_cat_cols + dataset_manager.label_num_cols:

        # determine prediction problem - regression or classification
        if label_col in dataset_manager.label_cat_cols:
            print("%s - categorical" % label_col)
            mode = "class"
            pos_label = "true"
        elif label_col in dataset_manager.label_num_cols:
            print("%s - numeric" % label_col)
            mode = "regr"

        target_df = target_df_[[dataset_manager.case_id_col, label_col]]
        train = train_.groupby(dataset_manager.case_id_col, as_index=False).apply(dataset_manager.add_target, target_df, label_col)
        test = test_.groupby(dataset_manager.case_id_col, as_index=False).apply(dataset_manager.add_target, target_df, label_col)

        # discard cases where a given regression_activity or a gateway (decision point) did not occur, i.e. target undefined
        train = train[train[label_col] != -1]
        test = test[test[label_col] != -1]

        # create prefix logs
        dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
        dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

        print(dt_train_prefixes.shape)
        print(dt_test_prefixes.shape)

        # extract arguments
        bucketer_args = {'encoding_method': bucket_encoding,
                         'case_id_col': dataset_manager.case_id_col,
                         'cat_cols': [dataset_manager.activity_col],
                         'num_cols': [],
                         'random_state': random_state}

        cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                            'static_cat_cols': dataset_manager.static_cat_cols,
                            'static_num_cols': dataset_manager.static_num_cols,
                            'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                            'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                            'fillna': fillna}

        # Bucketing prefixes based on control flow
        print("Bucketing prefixes...")
        bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)

        for i in range(n_iter):
            n_estimators = np.random.randint(40, 1000)
            learning_rate = np.random.uniform(0.01, 0.07)
            subsample = np.random.uniform(0.5, 1)
            max_depth = np.random.randint(3, 9)
            colsample_bytree = np.random.uniform(0.4, 1)
            min_child_weight = np.random.randint(1, 3)

            params = {'n_estimators': n_estimators,
                      'learning_rate': learning_rate,
                      'subsample': subsample,
                      'max_depth': max_depth,
                      'colsample_bytree': colsample_bytree,
                      'min_child_weight': min_child_weight,
                      'mode': mode,
                      'min_cases_for_training': n_min_cases_in_bucket}

            print("Cls params are: %s" % str(list(params.values())))
            pipelines = {}

            # train and fit pipeline for each bucket
            for bucket in set(bucket_assignments_train):
                print("Fitting pipeline for bucket %s..." % bucket)
                relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
                dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                               relevant_cases_bucket)  # one row per event
                train_y = dataset_manager.get_label(dt_train_bucket, label_col=label_col, mode=mode)

                feature_combiner = FeatureUnion(
                    [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
                pipelines[bucket] = Pipeline(
                    [('encoder', feature_combiner), ('cls', ClassifierFactory.get_classifier(cls_method, **params))])

                pipelines[bucket].fit(dt_train_bucket, train_y)

            # if the bucketing is prefix-length-based, then evaluate for each prefix length separately, otherwise evaluate all prefixes together
            max_evaluation_prefix_length = max_prefix_length if bucket_method == "prefix" else min_prefix_length

            prefix_lengths_test = dt_test_prefixes.groupby(dataset_manager.case_id_col).size()

            # test separately for each prefix length
            for nr_events in range(min_prefix_length, max_evaluation_prefix_length + 1):
                print("Predicting for %s events..." % nr_events)

                if bucket_method == "prefix":
                    # select only prefixes that are of length nr_events
                    relevant_cases_nr_events = prefix_lengths_test[prefix_lengths_test == nr_events].index

                    if len(relevant_cases_nr_events) == 0:
                        break

                    dt_test_nr_events = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes,
                                                                                     relevant_cases_nr_events)
                    del relevant_cases_nr_events
                else:
                    # evaluate on all prefixes
                    dt_test_nr_events = dt_test_prefixes.copy()

                # get predicted cluster for each test case
                bucket_assignments_test = bucketer.predict(dt_test_nr_events)

                # use appropriate classifier for each bucket of test cases
                # for evaluation, collect predictions from different buckets together
                preds = []
                test_y = []
                for bucket in set(bucket_assignments_test):
                    relevant_cases_bucket = dataset_manager.get_indexes(dt_test_nr_events)[
                        bucket_assignments_test == bucket]
                    dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_nr_events,
                                                                                  relevant_cases_bucket)  # one row per event

                    if len(relevant_cases_bucket) == 0:
                        continue

                    elif bucket not in pipelines:
                        # use mean value (in training set) as prediction
                        print("Bucket is not in pipeline, defaulting to averages")
                        avg_target_value = [np.mean(target_df_[label_col])] if mode == "regr" else [
                            dataset_manager.get_class_ratio2(target_df_, label_col=label_col)]
                        preds_bucket = avg_target_value * len(relevant_cases_bucket)

                    else:
                        # make actual predictions
                        preds_bucket = pipelines[bucket].predict_proba(dt_test_bucket)
                        # if only one class is present in predictions
                        if mode == "class" and pipelines[bucket]._final_estimator.hardcoded_prediction is not None:
                            tmp = np.zeros((len(preds_bucket), gateway_exits[label_col]))
                            tmp[:, preds_bucket[0]] = np.ones(len(preds_bucket))
                            preds_bucket = tmp

                    if mode == "regr":
                        # if cycle time is predicted to be negative, make it zero
                        preds_bucket = preds_bucket.clip(min=0)
                    elif preds_bucket.shape[1] != gateway_exits[label_col]:
                        # if some branches were not present in the training set, thus are never predicted
                        classes_as_is = pipelines[bucket]._final_estimator.cls.classes_
                        tmp = np.zeros((len(preds_bucket), gateway_exits[label_col]))
                        ii = 0
                        for class_to_be in np.arange(gateway_exits[label_col]):
                            if class_to_be in classes_as_is:
                                tmp[:, class_to_be] = preds_bucket[:, ii]
                                ii += 1
                        preds_bucket = tmp

                    preds.extend(preds_bucket)

                    # extract actual label values
                    test_y_bucket = dataset_manager.get_label(dt_test_bucket, label_col=label_col, mode=mode)  # one row per case
                    test_y.extend(test_y_bucket)

                score = {}
                if mode == "regr":
                    score["mae"] = mean_absolute_error(test_y, preds)
                elif len(set(test_y)) < 2:
                    score = {"logloss": None}
                else:
                    score["logloss"] = log_loss(test_y, preds)

                cls_params_str = ";".join([str(params[param]) for param in cls_params_names])

                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                label_col, method_name, cls_method, cls_params_str, nr_events, list(score)[0], list(score.values())[0]))
                # fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                # label_col, method_name, cls_method, cls_params_str, nr_events, list(score)[1], list(score.values())[1]))
