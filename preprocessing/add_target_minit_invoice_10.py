import os
from sys import argv
import numpy as np
import pandas as pd
from itertools import *
from tqdm import tqdm

from core.DatasetManager import DatasetManager

script_name = os.path.basename(argv[0]).split(".")[0]
dataset_ref = script_name.split("_", maxsplit=2)[-1]
home_dirs = os.environ['PYTHONPATH'].split(":")
home_dir = home_dirs[0]  # if there are multiple PYTHONPATHs, choose the first
logs_dir = "logdata/"

dataset_manager = DatasetManager(dataset_ref)
dtypes = {col: "str" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
          [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
    dtypes[col] = "float"


def add_cycle_times_gateway_classes(group):
    group = group.sort_values(dataset_manager.timestamp_col, ascending=True, kind='mergesort')
    group = group.reset_index(drop=True)
    for regression_activity in dataset_manager.label_num_cols:
        regression_activity_ids = group.index[group[dataset_manager.activity_col] == regression_activity]
        if regression_activity_ids.empty:
            continue
        cycle_time = 0
        for regression_activity_id in regression_activity_ids:
            tmp = group.loc[regression_activity_id][dataset_manager.timestamp_col] - group.loc[regression_activity_id - 1][dataset_manager.timestamp_col]
            tmp = tmp / np.timedelta64(1, 's')
            cycle_time = cycle_time + tmp
        group[regression_activity] = cycle_time / len(regression_activity_ids)  # if an activity is repeated multiple times, take an average of those

    # decision points based on directly-follow relations
    for index, row in islice(group.iterrows(), 0, len(group) - 1):
        if group.loc[index, dataset_manager.activity_col] == "Check_cost_center" and \
                group.loc[index + 1, dataset_manager.activity_col] == "Manual_identification_CC":
            group["x31"] = 0
        elif group.loc[index, dataset_manager.activity_col] == "Check_cost_center" and \
                group.loc[index + 1, dataset_manager.activity_col] == "Get_lowest_approval_level":
            group["x31"] = 1

    # for index, row in islice(group.iterrows(), 0, len(group) - 1):
    #     if group.loc[index, dataset_manager.activity_col] == "Check_whether_the_total_approval" and \
    #             group.loc[index + 1, dataset_manager.activity_col] == "Shift_to_higher_level":
    #         group["x41"] = 0
    #     elif group.loc[index, dataset_manager.activity_col] == "Check_whether_the_total_approval" and \
    #             group.loc[index + 1, dataset_manager.activity_col] != "Shift_to_higher_level":
    #         group["x41"] = 1

    # decision points based on presence/absence of specific activities
    if sum(group[dataset_manager.activity_col] == "Manual_enter_the_order_number") > 0:
        group["x11"] = 0
    else:
        group["x11"] = 1

    if sum(group[dataset_manager.activity_col] == "Check_cost_center") > 0 and sum(
                    group[dataset_manager.activity_col] == "Get_lowest_approval_level") > 0:
        group["x21"] = 1
    else:
        group["x21"] = 0

    # if sum(group[dataset_manager.activity_col] == "Manual_identification_CC") > 0:
    #     group["x31"] = 0
    # else:
    #     group["x31"] = 1

    if sum(group[dataset_manager.activity_col] == "Shift_to_higher_level") > 0:
        group["x41"] = 0
    elif sum(group[dataset_manager.activity_col] == "Get_lowest_approval_level") > 0:
        group["x41"] = 1

    if sum(group[dataset_manager.activity_col] == "Approving_on_specific_level") > 1 and sum(
            group[dataset_manager.activity_col] == "Check_whether_the_total_approval") > 1:
        group["x51"] = 1
    elif sum(group[dataset_manager.activity_col] == "Approving_on_specific_level") == 1 and sum(
            group[dataset_manager.activity_col] == "Check_whether_the_total_approval") == 1:
        group["x51"] = 0

    if sum(group[dataset_manager.activity_col] == "Status_change_to_Approved") > 0 and sum(
            group[dataset_manager.activity_col] == "Invoice_accounting") > 0:
        group["x61"] = 1
    else:
        group["x61"] = 0

    return group


data = pd.read_csv(os.path.join(home_dir, logs_dir, "%s.csv" % dataset_ref), sep=";", dtype=dtypes)
#data = data.head(102)
print(data[dataset_manager.case_id_col].nunique())
data[dataset_manager.timestamp_col] = pd.to_datetime(data[dataset_manager.timestamp_col])

for label_col in dataset_manager.label_cat_cols + dataset_manager.label_num_cols:
    data[label_col] = -1

# data["x51"] = 0  # some gateways (self-loops) include a path that is *always* taken, i.e. branches are not exclusive

tqdm.pandas()
data = data.groupby(dataset_manager.case_id_col).progress_apply(add_cycle_times_gateway_classes)
target = data.groupby(dataset_manager.case_id_col).head(n=1)
target = target[[dataset_manager.case_id_col] + dataset_manager.label_num_cols + dataset_manager.label_cat_cols]
target.to_csv(os.path.join(home_dir, logs_dir, "target/target_%s.csv" % dataset_ref), sep=",", index=False)
