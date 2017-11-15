import sys
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold

home_dirs = os.environ['PYTHONPATH'].split(":")
home_dir = home_dirs[0]
dataset_params_dir = os.path.join(home_dir, "core/dataset_params/")

class DatasetManager:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        dataset_params = pd.read_json(os.path.join(dataset_params_dir, "%s.json" % self.dataset_name), orient="index", typ="series")

        self.case_id_col = dataset_params[u'case_id_col']
        self.activity_col = dataset_params[u'activity_col']
        self.timestamp_col = dataset_params[u'timestamp_col']

        # possible names for label columns
        self.label_cat_cols = dataset_params[u'gateways']
        self.label_num_cols = dataset_params[u'regression_activities']

        # define features for predictions
        predictor_cols = ["dynamic_cat_cols", "static_cat_cols", "dynamic_num_cols", "static_num_cols"]
        for predictor_col in predictor_cols:
            for label in self.label_cat_cols + self.label_num_cols:
                if label in dataset_params[predictor_col]:
                    print("%s found in %s, it will be removed (not a feature)" % (label, predictor_col))
                    dataset_params[predictor_col].remove(label)  # exclude label attributes from features
            setattr(self, predictor_col, dataset_params[predictor_col])


    def add_remtime(self, group):
        group = group.sort_values(self.timestamp_col, ascending=True)
        end_date = group[self.timestamp_col].iloc[-1]
        tmp = end_date - group[self.timestamp_col]
        tmp = tmp.fillna(0)
        group["remtime"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds
        return group

    def add_target(self, group, target_df, label_col):
        this_case = group[self.case_id_col].iloc[0]
        group[label_col] = target_df.loc[target_df[self.case_id_col] == this_case, label_col].item()
        return group

    def get_median_case_duration(self, data):
        case_durations = data.groupby(self.case_id_col)['remtime'].max()
        return np.median(case_durations)

    def assign_label(self, group, threshold, label_col):
        group = group.sort_values(self.timestamp_col, ascending=True)
        case_duration = group["remtime"].iloc[0]
        group[label_col] = "false" if case_duration < threshold else "true"
        return group

    def split_data(self, data, train_ratio):
        # split into train and test using temporal split

        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(np.ceil(train_ratio*len(start_timestamps)))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')

        return (train, test)


    def generate_prefix_data(self, data, min_length, max_length):
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        dt_prefixes = data[data['case_length'] > min_length].groupby(self.case_id_col).head(min_length)  # strict inequality
        for nr_events in range(min_length+1, max_length+1):
            tmp = data[data['case_length'] > nr_events].groupby(self.case_id_col).head(nr_events)  # strict inequality
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
        dt_prefixes['case_length'] = dt_prefixes.groupby(self.case_id_col)[self.activity_col].transform(len)
        
        return dt_prefixes


    def get_pos_case_length_quantile(self, data, quantile=0.90):
        return int(np.ceil(data.groupby(self.case_id_col).size().quantile(quantile)))

    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data, label_col, mode="regr"):
        if mode == "regr":
            return data.groupby(self.case_id_col).min()[label_col]
        else:
            return data.groupby(self.case_id_col).first()[label_col]
    
    def get_class_ratio(self, data, label_col):
        class_freqs = data[label_col].value_counts()
        return class_freqs / class_freqs.sum()
    
    def get_stratified_split_generator(self, data, label_col, n_splits=5, shuffle=True, random_state=22, mode="regr"):
        grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
        skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state) if mode == "regr" else StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[label_col]):
            current_train_names = grouped_firsts[self.case_id_col][train_index]
            train_chunk = data[data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            test_chunk = data[~data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            yield (train_chunk, test_chunk)

    def evaluate_formula(self, formula, terms):
        for k, v in terms.items():
            exec("%s=%s" % (k, v))
        return eval(formula)
