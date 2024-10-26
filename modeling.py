import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression


layer_keys = ['layer_2', 'layer_15', 'layer_30']
hidden_keys = ["attention", "mlp"]
dataset = "qnota"
task="unanswerable"
hidden_states_dir = f"results/{task}/"

def get_pickles(files):
    subfiles = []
    for file in files:
        if ".pkl" in file:
            subfiles.append(int(file.split(".")[0]))
        else:
            pass
    return subfiles

def get_array(files, file_path):
    array = []
    for file in files:
        internal_array = []
        with open(f"{file_path}/{file}.pkl", "rb") as f:
            hidden_states_list = pickle.load(f)
        for hidden_states in hidden_states_list:
            for layer_key in layer_keys:
                for hidden_key in hidden_keys:
                        internal_array.append(hidden_states[layer_key][hidden_key][-1])
            array.append(internal_array)
    return np.array(array)


train_files = os.listdir(f"{hidden_states_dir}/train/{dataset}/")
train_files = get_pickles(train_files)
test_files = os.listdir(f"{hidden_states_dir}/test/{dataset}/")
test_files = get_pickles(test_files)
train_files = sorted(train_files)
test_files = sorted(test_files)
X_train = get_array(train_files, f"{hidden_states_dir}/train/{dataset}/")
X_test = get_array(test_files, f"{hidden_states_dir}/test/{dataset}/")
train_df = pd.read_csv(f"data/{task}/{dataset}_train.csv")
test_df = pd.read_csv(f"data/{task}/{dataset}_test.csv")
y_train = train_df["unanswerable"].values
y_test = test_df["unanswerable"].values

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print(f"Train score: {train_score}")
print(f"Test score: {test_score}")