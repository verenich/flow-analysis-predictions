from __future__ import division
import glob
import pandas as pd
import numpy as np
import os

input_data_folder = "../logdata/"
#filenames = glob.glob("*.csv")
#filenames = [filename for filename in filenames if os.path.getsize(filename) > 0]

filenames = ["hospital_billing.csv"]
timestamp_col = "Complete Timestamp" # column that indicates completion timestamp
case_id_col = "case_id"

def add_all_columns(group):
    group = group.sort_values(timestamp_col, ascending=True)
    group["event_nr"] = range(1,group.shape[0]+1)
    group["unique_timestamps"] = [len(group[timestamp_col].unique())] * len(group)
    group["total_timestamps"] = [len(group[timestamp_col])] * len(group)
    return group

with open("log_summary.csv", 'w') as fout:
    fout.write("%s,%s,%s,%s,%s,%s,%s\n" % (
    "log", "avg_unique_timestamp_per_trace", "avg_unique_timestamp_global",
    "cases_longer_5", "timestamp_0_after_5", "cases_longer_20", "timestamp_0_after_20"))
    for filename in filenames:
        #print(filename)
        # dtypes = {col:"str" for col in ["proctime", "elapsed", "label", "last"]} # prevent type coercion
        data = pd.read_csv(os.path.join(input_data_folder,filename), sep=";")
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        data = data.groupby(case_id_col).apply(add_all_columns)
        df0 = data.loc[data["event_nr"] == 1].copy()
        df0["UTR"] = df0["unique_timestamps"] / df0["total_timestamps"]
        #print("Avg percentage of unique timestamps per trace: %.3f" %np.mean(df0["UTR"]))
        #print("%s out of %s unique timestamps" %(len(data[timestamp_col].unique()),data[timestamp_col].count()))
        global_unique_timestamps = len(data[timestamp_col].unique()) / data[timestamp_col].count()
        cutoff = 5
        df1 = data.loc[data["event_nr"] == cutoff]
        #print("%s cases that reach length %d" %(df.shape[0],cutoff))
        #print("In %s of them elapsed time is still 0" %len(df.loc[df["elapsed"]==0]))
        cutoff = 20
        df2 = data.loc[data["event_nr"] == cutoff]
        #print("%s cases that reach length %d" %(df.shape[0],cutoff))
        fout.write("%s, %.3f, %.3f, %s, %s, %s, %s\n"%(filename, np.mean(df0["UTR"]), global_unique_timestamps, df1.shape[0],
                                 len(df1.loc[df1["timesincecasestart"] == 0]), df2.shape[0], len(df2.loc[df2["timesincecasestart"] == 0])))

