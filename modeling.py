import pandas as pd
import numpy as np
import os
import pickle


layer_keys = ['layer_2', 'layer_15', 'layer_30']
hidden_keys = ["attention", "mlp"]
dataset = "qnota"
hidden_states_dir = "results/unanswerable/"

def get_pickles(files):
    subfiles = []
    for file in files:
        if ".pkl" in file:
            subfiles.append(int(file.split(".")[0]))
        else:
            pass
    return subfiles


train_files = os.listdir(f"{hidden_states_dir}/train/{dataset}/")
train_files = get_pickles(train_files)
test_files = os.listdir(f"{hidden_states_dir}/test/{dataset}/")
test_files = get_pickles(test_files)
train_files = sorted(train_files)
test_files = sorted(test_files)

train_array = []
for file in train_files:
    internal_array = []
    with open(f"{hidden_states_dir}/train/{dataset}/{file}.pkl", "rb") as f:
        hidden_states_list = pickle.load(f)
    for hidden_states in hidden_states_list:
        for layer_key in layer_keys:
            for hidden_key in hidden_keys:
                for hidden_state in hidden_states[layer_key][hidden_key]:
                    internal_array.append(hidden_state)            
        train_array.append(internal_array)
train_array = np.array(train_array)

