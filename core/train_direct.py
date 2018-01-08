import os
import pickle
from sys import argv

import numpy as np
from numpy import array
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline, FeatureUnion

import BucketFactory
import ClassifierFactory
import EncoderFactory
from DatasetManager import DatasetManager

train_file = argv[1]
bucket_method = argv[2]
cls_encoding = argv[3]
cls_method = argv[4]
n_min_cases_in_bucket = int(argv[5])
label_col = "remtime"
mode = "regr"

dataset_ref = os.path.splitext(train_file)[0]
home_dirs = os.environ['PYTHONPATH'].split(":")
home_dir = home_dirs[0]  # if there are multiple PYTHONPATHs, choose the first
logs_dir = "logdata/"
training_params_dir = "core/training_params/"
results_dir = "results/validation/"
detailed_results_dir = "results/detailed/"
feature_importance_dir = "results/feature_importance/"
pickles_dir = "pkl/"

best_params = pd.read_json(os.path.join(home_dir, training_params_dir, "%s.json" % dataset_ref), typ="series")

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}

method_name = "%s_%s" % (bucket_method, cls_encoding)
methods = encoding_dict[cls_encoding]

outfile = os.path.join(home_dir, results_dir,
                       "validation_direct_%s_%s_%s_%s.csv" % (dataset_ref, method_name, cls_method, n_min_cases_in_bucket))
detailed_results_file = os.path.join(home_dir, detailed_results_dir,
                       "validation_direct_%s_%s_%s_%s.csv" % (dataset_ref, method_name, cls_method, n_min_cases_in_bucket))

pickle_file = os.path.join(home_dir, pickles_dir, '%s_%s_%s_%s_%s.pkl' % (dataset_ref, method_name, cls_method, label_col, n_min_cases_in_bucket))

random_state = 22
fillna = True
# n_min_cases_in_bucket = 100

##### MAIN PART ######
detailed_results = pd.DataFrame()
with open(outfile, 'w') as fout:
    fout.write("%s,%s,%s,%s,%s,%s,%s\n" % ("dataset", "method", "cls", "nr_events", "metric", "score", "nr_cases"))

    dataset_manager = DatasetManager(dataset_ref)
    dtypes = {col: "str" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
              [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
    for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
        dtypes[col] = "float"

    dtypes[label_col] = "float"  # remaining time should be float

    data = pd.read_csv(os.path.join(home_dir, logs_dir, train_file), sep=";", dtype=dtypes)
    #data = data.head(30000)
    data[dataset_manager.timestamp_col] = pd.to_datetime(data[dataset_manager.timestamp_col])

    # add label column to the dataset if it does not exist yet
    if label_col not in data.columns:
        print("column %s does not exist in the log, let's create it" % label_col)
        data = data.groupby(dataset_manager.case_id_col, as_index=False).apply(dataset_manager.add_remtime)

    # split data into training and test sets
    train, test = dataset_manager.split_data(data, train_ratio=0.8)
    # train = train.sort_values(dataset_manager.timestamp_col, ascending=True, kind='mergesort')

    # consider prefix lengths until 90th percentile of case length
    min_prefix_length = 2
    max_prefix_length = dataset_manager.max_prefix_length
    #max_prefix_length = min(6, dataset_manager.get_pos_case_length_quantile(data, 0.95))
    del data

    # create prefix logs
    dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

    print(dt_train_prefixes.shape)
    print(dt_test_prefixes.shape)


    model_training_required = True
    if model_training_required:

        # extract arguments
        bucketer_args = {'case_id_col': dataset_manager.case_id_col}

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

        pipelines = {}

        # train and fit pipeline for each bucket
        for bucket in set(bucket_assignments_train):
            print("Fitting pipeline for bucket %s..." % bucket)

            # set optimal params for this bucket
            if bucket_method == "prefix":
                cls_args = best_params[label_col][method_name][cls_method][u'%s' % bucket]
            else:
                cls_args = best_params[label_col][method_name][cls_method]
            cls_args['mode'] = mode
            cls_args['random_state'] = random_state
            cls_args['min_cases_for_training'] = n_min_cases_in_bucket

            # select relevant cases
            relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
            dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                           relevant_cases_bucket)  # one row per event
            train_y = dataset_manager.get_label(dt_train_bucket, label_col=label_col)

            feature_combiner = FeatureUnion(
                [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
            pipelines[bucket] = Pipeline(
                [('encoder', feature_combiner), ('cls', ClassifierFactory.get_classifier(cls_method, **cls_args))])

            pipelines[bucket].fit(dt_train_bucket, train_y)

            # feature_set = []
            # for feature_set_this_encoding in pipelines[bucket].steps[0][1].transformer_list:
            #     for feature in feature_set_this_encoding[1].columns.tolist():
            #         feature_set.append(feature)
            #
            # feats = {}  # a dict to hold feature_name: feature_importance
            # for feature, importance in zip(feature_set, pipelines[bucket].named_steps.cls.cls.feature_importances_):
            #     feats[feature] = importance  # add the name/value pair
            #
            # importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
            # importances = importances.sort_values(by='Gini-importance', ascending=False)
            # importances.to_csv(os.path.join(home_dir, feature_importance_dir, "feat_importance_%s_%s_%s_%s_%s.csv" %
            #                                 (dataset_ref, method_name, cls_method, label_col, bucket)))

        with open(pickle_file, 'wb') as f:
            pickle.dump(pipelines, f)
            pickle.dump(bucketer, f)


    prefix_lengths_test = dt_test_prefixes.groupby(dataset_manager.case_id_col).size()

    # test separately for each prefix length
    for nr_events in range(min_prefix_length, max_prefix_length + 1):
        print("Predicting for %s events..." % nr_events)

        # select only cases that are at least of length nr_events
        relevant_cases_nr_events = prefix_lengths_test[prefix_lengths_test == nr_events].index

        if len(relevant_cases_nr_events) == 0:
            break

        dt_test_nr_events = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_cases_nr_events)
        del relevant_cases_nr_events


        if not model_training_required:
            with open(pickle_file, 'rb') as f:
                pipelines = pickle.load(f)
                bucketer = pickle.load(f)

        # assign a bucket to each test case
        bucket_assignments_test = bucketer.predict(dt_test_nr_events)

        # use appropriate classifier for each bucket of test cases
        # for evaluation, collect predictions from different buckets together
        preds = []
        test_y = []
        for bucket in set(bucket_assignments_test):
            relevant_cases_bucket = dataset_manager.get_indexes(dt_test_nr_events)[bucket_assignments_test == bucket]
            dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_nr_events,
                                                                          relevant_cases_bucket)  # one row per event

            if len(relevant_cases_bucket) == 0:
                continue

            elif bucket not in pipelines:
                # use mean value (in training set) as prediction
                print("Bucket is not in pipeline, defaulting to averages")
                avg_target_value = [np.mean(train[label_col])]
                preds_bucket = array(avg_target_value * len(relevant_cases_bucket))

            else:
                # make actual predictions
                preds_bucket = pipelines[bucket].predict_proba(dt_test_bucket)

            preds_bucket = preds_bucket.clip(min=0)  # if remaining time is predicted to be negative, make it zero
            preds.extend(preds_bucket)

            # extract actual label values
            test_y_bucket = dataset_manager.get_label(dt_test_bucket, label_col=label_col)  # one row per case
            test_y.extend(test_y_bucket)

            case_ids = list(dt_test_bucket.groupby(dataset_manager.case_id_col).first().index)
            current_results = pd.DataFrame(
                {"method": method_name, "cls": cls_method, "nr_events": nr_events,
                 "predicted": preds_bucket, "actual": test_y_bucket, "case_id": case_ids})
            detailed_results = pd.concat([detailed_results, current_results], axis=0)

        score = {}
        if len(test_y) < 2:
            score = {"score1": 0, "score2": 0}
        else:
            score["mae"] = mean_absolute_error(test_y, preds)
            score["rmse"] = np.sqrt(mean_squared_error(test_y, preds))

        for k, v in score.items():
            fout.write("%s,%s,%s,%s,%s,%s,%s\n" % (dataset_ref, method_name, cls_method, nr_events, k, v, len(test_y)))

    print("\n")

detailed_results.to_csv(detailed_results_file, sep=";", index=False)
