import os
import pickle
from sys import argv

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

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
optimization_type = argv[6]  # FA - each activity separately, FA2 - all together ("remtime2")

dataset_ref = os.path.splitext(train_file)[0]
home_dirs = os.environ['PYTHONPATH'].split(":")
home_dir = home_dirs[0] # if there are multiple PYTHONPATHs, choose the first
logs_dir = "logdata/"
training_params_dir = "core/training_params/"
results_dir = "results/validation/"
detailed_results_dir = "results/detailed/"
feature_importance_dir = "results/feature_importance/"
pickles_dir = "pkl/"
formula_dir = "formulas/"

best_params = pd.read_json(os.path.join(home_dir, training_params_dir, "%s.json" % dataset_ref), typ="series")

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}

method_name = "%s_%s" % (bucket_method, cls_encoding)
methods = encoding_dict[cls_encoding]

outfile = os.path.join(home_dir, results_dir,
                       "validation_calibrated_%s_%s_%s_%s_%s.csv" % (optimization_type, dataset_ref, method_name, cls_method, n_min_cases_in_bucket))
detailed_results_file = os.path.join(home_dir, detailed_results_dir,
                       "validation_calibrated_%s_%s_%s_%s_%s.csv" % (optimization_type, dataset_ref, method_name, cls_method, n_min_cases_in_bucket))

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

    # if dataset_manager.mode == "regr":
    #     dtypes[dataset_manager.label_col] = "float"  # if regression, target value is float
    # else:
    #     dtypes[dataset_manager.label_col] = "str"  # if classification, preserve and do not interpret dtype of label

    data = pd.read_csv(os.path.join(home_dir, logs_dir, train_file), sep=";", dtype=dtypes)
    #data = data.head(30000)
    data[dataset_manager.timestamp_col] = pd.to_datetime(data[dataset_manager.timestamp_col])

    # split data into training and test sets
    train_, test = dataset_manager.split_data(data, train_ratio=0.8)
    # train = train.sort_values(dataset_manager.timestamp_col, ascending=True, kind='mergesort')

    # consider prefix lengths until 90th percentile of case length
    min_prefix_length = 2
    max_prefix_length = dataset_manager.max_prefix_length
    #max_prefix_length = min(6, dataset_manager.get_pos_case_length_quantile(data, 0.95))
    del data

    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

    # file with activities and gateways to predict
    target_df_ = pd.read_csv(os.path.join(home_dir, logs_dir, "target/target_%s" % train_file),
                            dtype={'%s' % dataset_manager.case_id_col: str})

    gateway_exits = {}
    for gateway in dataset_manager.label_cat_cols:
        gateway_exits[gateway] = target_df_.loc[target_df_[gateway] != -1][gateway].nunique()

    model_training_required = True
    if model_training_required:

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

            # discard cases where a given regression_activity or a gateway (decision point) did not occur, i.e. target undefined
            train = train[train[label_col] != -1]

            if label_col in dataset_manager.label_cat_cols:
                train, val = dataset_manager.split_data(train, train_ratio=0.8)
                dt_val_prefixes = dataset_manager.generate_prefix_data(val, min_prefix_length, max_prefix_length)
                print(dt_val_prefixes.shape)

            # create prefix logs
            dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
            #dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

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

            if label_col in dataset_manager.label_cat_cols:
                bucket_assignments_val = bucketer.fit_predict(dt_val_prefixes)

            if optimization_type == "FA":
                key_id = label_col
            elif optimization_type == "FA2":
                key_id = "remtime2"

            feature_combiner = {}
            classifier = {}

            # train and fit pipeline for each bucket
            for bucket in set(bucket_assignments_train):
                print("Fitting pipeline for bucket %s..." % bucket)

                # set optimal params for this bucket
                if bucket_method == "prefix":
                    cls_args = best_params[key_id][method_name][cls_method][u'%s' % bucket]
                else:
                    cls_args = best_params[key_id][method_name][cls_method]
                cls_args['mode'] = mode
                cls_args['random_state'] = random_state
                cls_args['min_cases_for_training'] = n_min_cases_in_bucket
                #print("Cls params are: %s" % str(list(cls_args.values())))

                # select relevant cases
                relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
                dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_cases_bucket)  # one row per event
                train_y = dataset_manager.get_label(dt_train_bucket, label_col=label_col, mode=mode)

                if label_col in dataset_manager.label_cat_cols:
                    relevant_cases_bucket_val = dataset_manager.get_indexes(dt_val_prefixes)[bucket_assignments_val == bucket]
                    dt_val_bucket = dataset_manager.get_relevant_data_by_indexes(dt_val_prefixes, relevant_cases_bucket_val)  # one row per event
                    val_y = dataset_manager.get_label(dt_val_bucket, label_col=label_col, mode=mode)
                    if len(set(train_y)) < 2 or len(set(val_y)) < 2:
                        break

                feature_combiner[bucket] = FeatureUnion(
                    [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
                X_train = feature_combiner[bucket].fit_transform(dt_train_bucket)

                if label_col in dataset_manager.label_cat_cols:
                    classifier[bucket] = xgb.XGBClassifier(**cls_args)
                    X_val = feature_combiner[bucket].fit_transform(dt_val_bucket)
                    eval_set = [(X_val, val_y)]
                    classifier[bucket].fit(X_train, train_y)
                    cal_method = "sigmoid" if X_val.shape[0] < 5000 else "isotonic"
                    classifier[bucket] = CalibratedClassifierCV(classifier[bucket], cv="prefit", method=cal_method)
                    classifier[bucket].fit(X_val, val_y)
                elif label_col in dataset_manager.label_num_cols:
                    classifier[bucket] = xgb.XGBRegressor(**cls_args)
                    classifier[bucket].fit(X_train, train_y)


            pickle_file = os.path.join(home_dir, pickles_dir,
                                       'calibrated_%s_%s_%s_%s_%s.pkl' % (dataset_ref, method_name, cls_method, label_col, n_min_cases_in_bucket))
            with open(pickle_file, 'wb') as f:
                pickle.dump(classifier, f)
                pickle.dump(bucketer, f)
                pickle.dump(feature_combiner, f)


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
        result = pd.DataFrame()

        if dataset_ref == "BPI2012W" or dataset_ref == "BPI2012W_no_dup":
            current_prob_index = 5
        elif dataset_ref == "helpdesk":
            current_prob_index = 3
        else:
            current_prob_index = 1

        # add probabilities of dead and phantom branches
        probs_to_add = None
        if current_prob_index > 1:
            probs_to_add = range(1, current_prob_index)

        for label_col in dataset_manager.label_num_cols + dataset_manager.label_cat_cols:
            if label_col in dataset_manager.label_cat_cols:
                print("Predicting %s" % label_col)
                mode = "class"
                pos_label = "true"
            elif label_col in dataset_manager.label_num_cols:
                print("Predicting %s" % label_col)
                mode = "regr"
            pickle_file = os.path.join(home_dir, pickles_dir,
                                       'calibrated_%s_%s_%s_%s_%s.pkl' % (dataset_ref, method_name, cls_method, label_col, n_min_cases_in_bucket))

            with open(pickle_file, 'rb') as f:
                classifier = pickle.load(f)
                bucketer = pickle.load(f)
                feature_combiner = pickle.load(f)


            # get predicted cluster for each test case
            bucket_assignments_test = bucketer.predict(dt_test_nr_events)

            # use appropriate classifier for each bucket of test cases
            # for evaluation, collect predictions from different buckets together
            preds = []
            case_ids = []
            for bucket in set(bucket_assignments_test):
                relevant_cases_bucket = dataset_manager.get_indexes(dt_test_nr_events)[bucket_assignments_test == bucket]
                dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_nr_events,
                                                                              relevant_cases_bucket)  # one row per event

                if len(relevant_cases_bucket) == 0:
                    continue

                elif bucket not in classifier:
                    # regression - use mean value (in training set) as prediction
                    # classification - use the historical class ratio
                    print("Bucket is not in pipeline, defaulting to averages")
                    avg_target_value = [np.mean(target_df_[label_col])] if mode == "regr" else [
                        dataset_manager.get_class_ratio(target_df_, label_col=label_col)]
                    preds_bucket = avg_target_value * len(relevant_cases_bucket)

                else:
                    # make actual predictions
                    X_test = feature_combiner[bucket].transform(dt_test_bucket)
                    preds_bucket = classifier[bucket].predict_proba(X_test) if mode == "class" else classifier[bucket].predict(X_test)
                    # if mode == "class" and classifier[bucket]._final_estimator.hardcoded_prediction is not None:
                    #     preds_bucket2 = np.zeros((len(preds_bucket), gateway_exits[label_col]))
                    #     preds_bucket2[:,preds_bucket[0]] = np.ones(len(preds_bucket))
                    #     preds_bucket = preds_bucket2
                    case_ids_bucket = dataset_manager.get_indexes(dt_test_bucket)

                if mode == "regr":
                    # if cycle time is predicted to be negative, make it zero
                    preds_bucket = preds_bucket.clip(min=0)
                elif preds_bucket.shape[1] != gateway_exits[label_col]:
                    # if some branches were not present in the training set, thus are never predicted
                    # if classifier[bucket]._final_estimator.mean_prediction is not None:
                    #     classes_as_is = classifier[bucket]._final_estimator.mean_prediction.index
                    # else:
                    # classes_as_is = classifier[bucket]._final_estimator.cls.classes_
                    classes_as_is = classifier[bucket].base_estimator.classes_
                    preds_bucket2 = np.zeros((len(preds_bucket), gateway_exits[label_col]))
                    ii = 0
                    for class_to_be in np.arange(gateway_exits[label_col]):
                        if class_to_be in classes_as_is:
                            preds_bucket2[:, class_to_be] = preds_bucket[:,ii]
                            ii+=1
                    preds_bucket = preds_bucket2

                preds.extend(preds_bucket)
                case_ids.extend(case_ids_bucket)

            preds = np.array(preds)  # convert list of arrays to 2D array
            case_ids = [i.split('_')[0] for i in case_ids]
            if mode == "regr":
                predicted = pd.Series(preds)
                predicted.index = case_ids
                result[label_col] = predicted

            elif mode == "class":
                for column in preds.T:
                    predicted = pd.Series(column)
                    predicted.index = case_ids
                    result["p%s"%current_prob_index] = predicted
                    current_prob_index += 1

        if probs_to_add is not None:
            for prob_to_add in probs_to_add:
                result["p%s" % prob_to_add] = 1

        # read in the file with test cases and formulae
        formula_file = os.path.join(home_dir, formula_dir, '%s/test_len_%s.xes_formula.csv' % (dataset_ref, nr_events))
        testformula = pd.read_csv(formula_file, sep=";", header=None, usecols=[0,1], dtype=str)
        testformula.columns = [dataset_manager.case_id_col, 'formula']
        testformula.index = testformula[dataset_manager.case_id_col]

        result = pd.merge(result, testformula, left_index=True, right_index=True)
        result["predicted"] = -1

        terms = {}  # dictionary to store formula terms
        for index, row in result.iterrows():
            for vn in list(result.columns)[:-3]:
                terms[vn] = row[vn]
                #locals()[vn] = row[vn]
            result.loc[index, "predicted"] = dataset_manager.evaluate_formula(row["formula"], terms)


        # extract actual label values
        test_y = dataset_manager.get_label(dt_test_nr_events, label_col = "remtime")  # one row per case
        test_y = pd.DataFrame({dataset_manager.case_id_col: test_y.index, 'remtime': test_y.values})
        test_y[dataset_manager.case_id_col] = test_y[dataset_manager.case_id_col].apply(lambda x: x.split("_")[0])
        result = pd.merge(result, test_y, on=dataset_manager.case_id_col)

        result.loc[result["predicted"]<0, "predicted"] = 0  # if remaining time is predicted to be negative, make it zero
        current_results = pd.DataFrame(
            {"method": method_name, "cls": cls_method, "nr_events": nr_events, "predicted": result["predicted"],
             "actual": result["remtime"], "case_id": result[dataset_manager.case_id_col]})
        detailed_results = pd.concat([detailed_results, current_results], axis=0)

        score = {}
        if len(test_y) < 2:
            score = {"score1": 0, "score2": 0}
        else:
            score["mae"] = mean_absolute_error(result["predicted"], result["remtime"])
            score["rmse"] = np.sqrt(mean_squared_error(result["predicted"], result["remtime"]))

        fout.write("%s,%s,%s,%s,%s,%s,%s\n" % (dataset_ref, method_name, cls_method, nr_events,
                                            list(score)[0], list(score.values())[0], len(result["remtime"])))
        fout.write("%s,%s,%s,%s,%s,%s,%s\n" % (dataset_ref, method_name, cls_method, nr_events,
                                            list(score)[1], list(score.values())[1], len(result["remtime"])))

    print("\n")

detailed_results.to_csv(detailed_results_file, sep=";", index=False)
