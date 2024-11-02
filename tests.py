# Write tests for:
# Modeling of probes on trivial instance
# Modeling of probes on non-trivial instance
# -------------------------------------------
import unittest
from warnings import warn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from compute_hidden import *
from data import *
import torch
import numpy as np
import os
import shutil
import pandas as pd



model_name = "meta-llama/Llama-3.1-8B-Instruct"
tracking_mlp_pre_residual = True
results_dir=os.environ["RESULTS_DIR"]+"/tests"
if not os.path.exists(results_dir):
    os.makedirs(results_dir+"/train/testing")
    os.makedirs(results_dir+"/test/testing")

train_save_folder = results_dir+"/train/testing"
test_save_folder = results_dir+"/test/testing"

def clear_save_cache():
    if os.path.exists(train_save_folder):
        shutil.rmtree(train_save_folder)
    if os.path.exists(test_save_folder):
        shutil.rmtree(test_save_folder)
    os.makedirs(results_dir+"/train/testing")
    os.makedirs(results_dir+"/test/testing")

eps = 1e-7

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def diagnostic_difference(a, b):
    print(f"{a.shape}, {b.shape}")
    print(f"Mean Comparison: {a.mean()}, {b.mean()}")
    print(f"Max Comparison: {a.max()}, {b.max()}")
    print(f"Min Comparison: {a.min()}, {b.min()}")
    print(f"Sum Comparison: {a.sum()}, {b.sum()}")
    print(f"Std Comparison: {a.std()}, {b.std()}")
    print(f"Mean Diff Comparison: {np.abs(a-b).mean()}")
    print(f"Max Diff Comparison: {np.abs(a-b).max()}")
    print(f"Min Diff Comparison: {np.abs(a-b).min()}")
    print(f"Std Diff Comparison: {np.abs(a-b).std()}")

def shape_equal(a, b):
    return a.shape == b.shape

def array_equal(a, b):
    return np.abs((a-b)).max() < eps

def compute_mlp_correct(prompts, track_layers):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    set_tracking_config(config, track_layers=track_layers, track_mlp=True, track_attention=True, track_projection=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    corrects_all = []
    hidden_states_all = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=5, stop_strings=["[STOP]"], pad_token_id=tokenizer.eos_token_id, tokenizer=tokenizer, output_hidden_states=True, output_attentions=True, return_dict_in_generate=True)
        # output.hidden_states is tuple of length max_new_tokens with each element being a tuple of length num_layers=33, with each element of that being a tensor of shape (1, seq_length, hidden_size)
        # for output.hidden_states[0], the shape of all tensors is (1, inputs.input_ids.shape[1], hidden_size)
        # for all other output.hidden_states[i] the shape is (1, 1, hidden_size)
        corrects = {}
        for layer in range(len(output.hidden_states[0])):
            corrects[f"layer_{layer}"] = {}
            for key in ["mlp"]:
                tensors = [output.hidden_states[0][layer]]
                for i in range(1, len(output.hidden_states)):
                    next_tensor = output.hidden_states[i][layer]
                    tensors.append(next_tensor)
                tensors = torch.cat(tensors, dim=1)
                corrects[f"layer_{layer}"][key] = to_numpy(tensors)[0]
        hidden_states = read_hidden_states(model.probe_hidden_output)
        model.probe_reset_hidden_output()
        corrects_all.append(corrects)
        hidden_states_all.append(hidden_states)
    return corrects_all, hidden_states_all



class TestHiddenStates(unittest.TestCase):
    def compare_hidden_state_lists(self, a, b, check_layers, check_keys):
        self.assertTrue(len(a) == len(b))
        for i in range(len(a)):
            for layer in check_layers:
                for key in check_keys:
                    self.assertTrue(f"layer_{layer}" in a[i])
                    self.assertTrue(f"layer_{layer}" in b[i])
                    self.assertTrue(key in a[i][f"layer_{layer}"])
                    self.assertTrue(key in b[i][f"layer_{layer}"])                
                    a_array = a[i][f"layer_{layer}"][key]
                    b_array = b[i][f"layer_{layer}"][key]
                    self.assertTrue(shape_equal(a_array, b_array))
                    self.assertTrue(array_equal(a_array, b_array))
        return

    def test_compute_hidden(self):
        if tracking_mlp_pre_residual:
            return
        track_layers = [2, 15, 30]
        prompt = "There is no "
        corrects_all, hidden_states_all = compute_mlp_correct([prompt], track_layers)
        self.compare_hidden_state_lists(corrects_all, hidden_states_all, track_layers, ["mlp"])
        return
    
    def test_save_load_hidden(self):
        track_layers = [2, 15, 30]
        prompts = ["There is no ", "The quick brown ."]
        corrects_all, hidden_states_all = compute_mlp_correct(prompts, track_layers)
        alt_save_hidden_states(hidden_states_all, train_save_folder, 1, exists_ok=True)
        save_hidden_states(hidden_states_all, f"{test_save_folder}/test_hidden.pkl")
        alt_loaded_hidden_states, indices = alt_load_hidden_states(train_save_folder, state_processor=null_state_processor)
        loaded_hidden_states = load_hidden_states(f"{test_save_folder}/test_hidden.pkl")
        self.compare_hidden_state_lists(alt_loaded_hidden_states, hidden_states_all, track_layers, ["mlp", "attention", "projection"])
        self.compare_hidden_state_lists(loaded_hidden_states, hidden_states_all, track_layers, ["mlp", "attention", "projection"])
        self.compare_hidden_state_lists(alt_loaded_hidden_states, loaded_hidden_states, track_layers, ["mlp", "attention", "projection"])
        if not tracking_mlp_pre_residual:
            self.compare_hidden_state_lists(corrects_all, hidden_states_all, track_layers, ["mlp"])  # Sanity
        clear_save_cache()
        return
    
class TestDataFunctions(unittest.TestCase):
    def basic_test(self, df):
        for column in ["idx", "text"]:
            self.assertTrue(column in df.columns)
        either = "label" in df.columns or "gold" in df.columns
        self.assertTrue(either)
        return
    
    def df_equal(self, a, b):
        self.assertTrue(set(a.columns) == set(b.columns))
        self.assertTrue(len(a) == len(b))
        for column in a.columns:
            if not (a[column] == b[column]).all():
                return False
        return True

    def test_setup_data(self):
        setup_fns = [ToxicityAvoidance.setup_toxic_chat, ToxicityAvoidance.setup_real_toxicity_prompts, 
                     Jailbreak.setup_toxic_chat, Unanswerable.setuphealthver, Unanswerable.setupqnota, 
                     Unanswerable.setupselfaware, Unanswerable.setupsquad, Unanswerable.setupknown_unknown, 
                     Confidence.setupmmlu]
        for setup_fn in setup_fns:
            train, test, dataset_name = setup_fn(save=False)
            print(f"Dataset: {dataset_name}")
            self.basic_test(train)
            self.basic_test(test)
        return
    
    def test_randomization(self):
        process_fns = [process_selfaware]
        setup_fns = [Unanswerable.setupqnota, Unanswerable.setupsquad, Confidence.setupmmlu]
        for process_fn in process_fns:
            name = process_fn.__name__
            print(f"Processing: {name}")
            train1, test1 = process_fn(save=False, random_seed=42)
            train2, test2 = process_fn(save=False, random_seed=42)
            train3, test3 = process_fn(save=False, random_seed=43)
            self.assertTrue(self.df_equal(train1, train2))
            self.assertTrue(self.df_equal(test1, test2))
            self.assertFalse(self.df_equal(train1, train3))
        
        for setup_fn in setup_fns:
            train1, test1, dataset_name = setup_fn(save=False, random_seed=42)
            train2, test2, dataset_name = setup_fn(save=False, random_seed=42)
            train3, test3, dataset_name = setup_fn(save=False, random_seed=43)
            print(f"Dataset: {dataset_name}")
            self.assertTrue(self.df_equal(train1, train2))
            self.assertTrue(self.df_equal(test1, test2))
            self.assertFalse(self.df_equal(train1, train3))

        return



if __name__ == "__main__":
    TestHiddenStates().test_compute_hidden()