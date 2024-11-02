# Write tests for:
# Saving of hidden states
# Loading of hidden states
# Modeling of probes on trivial instance
# Modeling of probes on non-trivial instance
# -------------------------------------------
import unittest
from warnings import warn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from compute_hidden import set_tracking_config, read_hidden_states, save_hidden_states, alt_save_hidden_states, load_hidden_states, alt_load_hidden_states
import torch
import numpy as np

model_name = "meta-llama/Llama-3.1-8B-Instruct"

eps = 1e-5

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def shape_equal(a, b):
    return a.shape == b.shape

def array_equal(a, b):
    return np.abs((a-b)).max() < eps

def compute_mlp_correct(prompt, track_layers):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    set_tracking_config(config, track_layers=track_layers, track_mlp=True, track_attention=False, track_projection=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=5, stop_strings=["[STOP]"], pad_token_id=tokenizer.eos_token_id, tokenizer=tokenizer, output_hidden_states=True, output_attentions=True, return_dict_in_generate=True)
    # output.hidden_states is tuple of length max_new_tokens with each element being a tuple of length num_layers=33, with each element of that being a tensor of shape (1, seq_length, hidden_size)
    # for output.hidden_states[0], the shape of all tensors is (1, inputs.input_ids.shape[1], hidden_size)
    # for all other output.hidden_states[i] the shape is (1, 1, hidden_size)
    corrects = {}
    for layer in track_layers:
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
    return corrects, hidden_states


class TestHiddenStates(unittest.TestCase):
    def test_compute_hidden(self):
        warn("If we are tracking the MLP before adding it to the residual stream, then this test will fail")
        track_layers_list = [[2, 15, 30], [0, 4, 28]]
        for track_layers in track_layers_list:
            prompt = "There is no "
            corrects, hidden_states = compute_mlp_correct(prompt, track_layers)
            for key in corrects:
                for subkey in corrects[key]:
                    self.assertTrue(shape_equal(corrects[key][subkey], hidden_states[key][subkey]))
                    self.assertTrue(array_equal(corrects[key][subkey], hidden_states[key][subkey]))


if __name__ == "__main__":
    unittest.main()