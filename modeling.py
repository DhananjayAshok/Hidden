import pandas as pd
import numpy as np
import os
import pickle
import warnings
from sklearn.linear_model import LogisticRegression


layer_keys = ['layer_2', 'layer_15', 'layer_30']
hidden_keys = ["attention", "mlp"]
dataset = "mmlu"
task="confidence"
results_dir = os.getenv("RESULTS_DIR")
data_dir = os.getenv("DATA_DIR")
hidden_states_dir = f"{results_dir}/{task}/"
read_from_results_dir = True if task in ["confidence"] else False

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
    drop_indexes = []
    for i, file in enumerate(files):
        try:
            with open(f"{file_path}/{file}.pkl", "rb") as f:
                hidden_states_list = pickle.load(f)
            for hidden_states in hidden_states_list:
                internal_array = []
                for layer_key in layer_keys:
                    for hidden_key in hidden_keys:
                        internal_array.extend(hidden_states[layer_key][hidden_key][-1])
                array.append(internal_array)
        except OSError:
            warnings.warn(f"File {file_path}/{file}.pkl Bad Address or File Not Found")
            if i < len(files) - 1:
                next_file = files[i + 1]
                drop_indexes.extend(list(range(file, next_file))) # Should this be inclusive?
            else:
                raise ValueError(f"File error on the last file. Not Implemented Yet")
    return np.array(array), drop_indexes


train_files = os.listdir(f"{hidden_states_dir}/train/{dataset}/")
train_files = get_pickles(train_files)
test_files = os.listdir(f"{hidden_states_dir}/test/{dataset}/")
test_files = get_pickles(test_files)
train_files = sorted(train_files)
test_files = sorted(test_files)
X_train, train_drop_indices = get_array(train_files, f"{hidden_states_dir}/train/{dataset}/")
X_test, test_drop_indices = get_array(test_files, f"{hidden_states_dir}/test/{dataset}/")

if read_from_results_dir:
    train_df = pd.read_csv(f"{results_dir}/{task}/{dataset}_train_inference.csv")
    test_df = pd.read_csv(f"{results_dir}/{task}/{dataset}_test_inference.csv")
else:
    train_df = pd.read_csv(f"{data_dir}/{task}/{dataset}_train.csv")
    test_df = pd.read_csv(f"{data_dir}/{task}/{dataset}_test.csv")

train_df = train_df.drop(train_drop_indices)
test_df = test_df.drop(test_drop_indices)
breakpoint()

if task == "unanswerable":
    y_train = train_df["unanswerable"].values
    y_test = test_df["unanswerable"].values
elif task == "confidence":
    y_train = train_df["correct"].values
    y_test = test_df["correct"].values
    nans = train_df["output_parsed"].isna()
    train_df = train_df[~nans]
    y_train = y_train[~nans]
    X_train = X_train[~nans]
    nans = test_df["output_parsed"].isna()
    test_df = test_df[~nans]
    y_test = y_test[~nans]
    X_test = X_test[~nans]

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print(f"Base rate: {y_train.mean()} (Train), {y_test.mean()} (Test)")
print(f"Train score: {train_score}")
print(f"Test score: {test_score}")