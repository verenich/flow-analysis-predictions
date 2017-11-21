# coding: utf-8

import pandas as pd
import numpy as np
import os
import sys

input_data_folder = "../logdata/orig/"
output_data_folder = "../logdata/"
in_filename = "Invoice Approval.csv"

case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "Complete Timestamp"

category_freq_threshold = 10

# features for classifier
dynamic_cat_cols = ["Activity", 'Resource', 'ActivityFinalAction', "EventType"]
static_cat_cols = ["CostCenter.Code", "Supplier.City", "Supplier.Name", "Supplier.State"]
dynamic_num_cols = []
static_num_cols = ["InvoiceTotalAmountWithoutVAT"]

static_cols = static_cat_cols + static_num_cols + [case_id_col]
dynamic_cols = dynamic_cat_cols + dynamic_num_cols + [timestamp_col]
cat_cols = dynamic_cat_cols + static_cat_cols


def extract_timestamp_features(group):
    group = group.sort_values(timestamp_col, ascending=False, kind='mergesort')

    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    tmp = tmp.fillna(0)
    group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 's')))  # m is for minutes

    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    tmp = tmp.fillna(0)
    group["timesincecasestart"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 's')))  # m is for minutes

    tmp = group[timestamp_col].iloc[0] - group[timestamp_col]
    tmp = tmp.fillna(0)
    group["remtime"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 's')))

    group = group.sort_values(timestamp_col, ascending=True, kind='mergesort')
    group["event_nr"] = range(1, len(group) + 1)

    return group


def check_if_any_of_activities_exist(group, activities):
    if np.sum(group[activity_col].isin(activities)) > 0:
        return True
    else:
        return False


data = pd.read_csv(os.path.join(input_data_folder, in_filename), sep=",")
data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)

# discard incomplete cases - completed cases should terminate with "Process end" activity
last_events = data.sort_values([timestamp_col], ascending=True, kind='mergesort').groupby(case_id_col).last()[
    "Activity"]
complete_cases = last_events.index[last_events == "Process end"]
data = data[data[case_id_col].isin(complete_cases)]

data = data[static_cols + dynamic_cols]

# add features extracted from timestamp
data[timestamp_col] = pd.to_datetime(data[timestamp_col])
data["month"] = data[timestamp_col].dt.month
data["weekday"] = data[timestamp_col].dt.weekday
data["hour"] = data[timestamp_col].dt.hour
data = data.groupby(case_id_col).apply(extract_timestamp_features)

# add inter-case features
print("Extracting open cases...")
sys.stdout.flush()
data = data.sort_values([timestamp_col], ascending=True, kind='mergesort')
dt_first_last_timestamps = data.groupby(case_id_col)[timestamp_col].agg([min, max])
dt_first_last_timestamps.columns = ["start_time", "end_time"]
# data["open_cases"] = data[timestamp_col].apply(get_open_cases)
case_end_times = dt_first_last_timestamps.to_dict()["end_time"]

data["open_cases"] = 0
case_dict_state = {}
for idx, row in data.iterrows():
    case = row[case_id_col]
    current_ts = row[timestamp_col]

    # save the state
    data.set_value(idx, 'open_cases', len(case_dict_state))

    if current_ts >= case_end_times[case]:
        if case in case_dict_state:
            del case_dict_state[case]
    else:
        case_dict_state[case] = 1

print("Imputing missing values...")
sys.stdout.flush()
# impute missing values
grouped = data.sort_values(timestamp_col, ascending=True, kind='mergesort').groupby(case_id_col)
for col in static_cols + dynamic_cols:
    data[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))

data[cat_cols] = data[cat_cols].fillna('missing')
data = data.fillna(0)

# set infrequent factor levels to "other"
for col in cat_cols:
    counts = data[col].value_counts()
    mask = data[col].isin(counts[counts >= category_freq_threshold].index)
    data.loc[~mask, col] = "other"

# remove spaces from activity names
data[activity_col].replace('\s+', '_', regex=True, inplace=True)

# remove traces shorter than 2 events
'''
def get_case_length(group):
    group['case_length'] = group.shape[0]
    return group

data = data.groupby(case_id_col).apply(get_case_length)
data = data[data["case_length"] > 2]
data.groupby("case_length")[case_id_col].nunique()
data = data.drop("case_length", axis=1)
'''

data.to_csv(os.path.join(output_data_folder, "minit_invoice.csv"), sep=";", index=False)
