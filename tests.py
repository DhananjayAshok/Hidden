# Write tests for:
# Saving of hidden states
# Loading of hidden states
# Modeling of probes on trivial instance
# Modeling of probes on non-trivial instance
# -------------------------------------------
import unittest
from warnings import warn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from compute_hidden import *
import torch
import numpy as np
import os

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tracking_mlp_pre_residual = True
results_dir=os.environ("RESULTS_DIR")+"/tests"
if not os.path.exists(results_dir):
    os.makedirs(results_dir+"/train/testing")
    os.makedirs(results_dir+"/test/testing")

train_save_folder = results_dir+"/train/testing"
test_save_folder = results_dir+"/test/testing"

def clear_save_cache():
    if os.path.exists(train_save_folder):
        os.rmdir(train_save_folder)
    if os.path.exists(test_save_folder):
        os.rmdir(test_save_folder)
    os.makedirs(results_dir+"/train/testing")
    os.makedirs(results_dir+"/test/testing")

eps = 1e-5

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def diagnostic_difference(a, b):
    print(f"{a.shape}, {b.shape}")
    print(f"Mean Comparison: {a.mean()}, {b.mean()}")
    print(f"Max Comparison: {a.max()}, {b.max()}")
    print(f"Min Comparison: {a.min()}, {b.min()}")
    print(f"Sum Comparison: {a.sum()}, {b.sum()}")
    print(f"Std Comparison: {np.abs(a-b).max()}")
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
    set_tracking_config(config, track_layers=track_layers, track_mlp=True, track_attention=False, track_projection=False)
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
    def test_compute_hidden(self):
        if tracking_mlp_pre_residual:
            return
        track_layers = [2, 15, 30]
        prompt = "There is no "
        corrects_all, hidden_states_all = compute_mlp_correct([prompt], track_layers)
        for i in range(len(corrects_all)):
            corrects = corrects_all[i]
            hidden_states = hidden_states_all[i]
            for hidden_layer in hidden_states:
                hidden_array = hidden_states[hidden_layer]["mlp"]
                correct_array = corrects[hidden_layer]["mlp"]
                self.assertTrue(shape_equal(hidden_array, correct_array))
                self.assertTrue(array_equal(hidden_array, correct_array))
        del corrects_all, hidden_states_all
        torch.cuda.empty_cache()
        return
    
    def test_save_load_hidden(self):
        track_layers = [2, 15, 30]
        prompts = ["There is no ", "The quick brown ."]
        corrects_all, hidden_states_all = compute_mlp_correct(prompts, track_layers)
        alt_save_hidden_states(hidden_states_all, train_save_folder, 0, exists_ok=True)
        save_hidden_states(hidden_states_all, f"{test_save_folder}/test_hidden.pkl")
        alt_loaded_hidden_states = alt_load_hidden_states(train_save_folder, state_processor=null_state_processor)
        loaded_hidden_states = load_hidden_states(f"{test_save_folder}/test_hidden.pkl")
        breakpoint()




    




if __name__ == "__main__":
    unittest.main()