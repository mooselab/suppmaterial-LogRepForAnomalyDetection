# Acknowledgement:
# Some of the codes are adapted from Loglizer project(https://github.com/logpai/loglizer). 

import numpy as np
import pandas as pd
import os
from collections import Counter

data_dir = "./dataset/BGL"
log_name = "BGL_full.log_structured_0.8_rad_1.0"
dataset_file = os.path.join(data_dir,log_name+'.npz')
dataset = np.load(dataset_file, allow_pickle=True)

x_train = dataset["x_train"][()]
y_train = dataset["y_train"]
x_test = dataset["x_test"][()]
y_test = dataset["y_test"]

train_anomaly = 100 * sum(y_train) / len(y_train)
test_anomaly = 100 * sum(y_test) / len(y_test)

print("# train sessions: {} ({:.2f}%)".format(len(x_train), train_anomaly))
print("# test sessions: {} ({:.2f}%)".format(len(x_test), test_anomaly))

def getEventSeqNew(restructured_logs):
    blk_events = {}
    for blk, eventList in restructured_logs.items():
        for event in eventList:
            if blk not in blk_events:
                blk_events[blk] = [event["EventId"]]
            else:
                blk_events[blk].append(event["EventId"])
    return blk_events

def transform_train_data(X_seq):
    X_counts = []
    for i in range(X_seq.shape[0]):
        event_counts = Counter(X_seq[i])
        X_counts.append(event_counts)
    X_df = pd.DataFrame(X_counts)
    X_df = X_df.fillna(0)
    events = X_df.columns
    X = X_df.values
    return (X, events)

def transform_test_data(X_seq, events):
    X_counts = []
    for i in range(X_seq.shape[0]):
        event_counts = Counter(X_seq[i])
        X_counts.append(event_counts)
    X_df = pd.DataFrame(X_counts)
    X_df = X_df.fillna(0)
    # treat the counts of the missing events as 0s
    empty_events = set(events) - set(X_df.columns)
    for event in empty_events:
        X_df[event] = [0] * len(X_df)
    X = X_df[events].values
    return X

def MergeDict(dict1, dict2):
    res = {**dict1, **dict2}
    return res



len_train = len(y_train)
len_test = len(y_test)
print(len_train, len_test)

x_train_seq = getEventSeqNew(x_train)
x_test_seq = getEventSeqNew(x_test)

x_train_evt_seq = []
x_test_evt_seq = []
for key, seq in x_train_seq.items():
    x_train_evt_seq.append(seq)

for key, seq in x_test_seq.items():
    x_test_evt_seq.append(seq)
    
x_train_evt_seq = np.array(x_train_evt_seq, dtype=object)
x_test_evt_seq = np.array(x_test_evt_seq, dtype=object)

x_train_transformed = transform_train_data(x_train_evt_seq)
x_train = x_train_transformed[0]

events = x_train_transformed[1]

x_test = transform_test_data(x_test_evt_seq, events)

save_path = os.path.join(data_dir,'MCV_'+log_name+'.npz')
np.savez(save_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)