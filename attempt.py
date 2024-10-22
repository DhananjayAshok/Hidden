from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import warnings
import numpy as np
import torch
from tqdm import tqdm
import pickle


def set_tracking_config(config, track_layers=[2, 15, 30], track_mlp=True, track_attention=True):
    config.track_layers = track_layers
    config.track_mlp = track_mlp
    config.track_attention = track_attention
    return
     
     

def detect_hidden_states_parameters(single_token_probe_hidden_output):
    assert isinstance(single_token_probe_hidden_output, dict)
    layers = list(single_token_probe_hidden_output.keys())
    assert len(layers) > 0
    input_length = None
    batch_size = None
    keys = []
    layer = layers[0]
    if len(single_token_probe_hidden_output[layer]) == 0:
                warnings.warn("Improper initialization (layer has no keys)")
                return None
    for key in single_token_probe_hidden_output[layer]:
        if batch_size is None or input_length is None:
            value = single_token_probe_hidden_output[layer][key]
            if value is None:
                warnings.warn("Trying to read hidden states before they are set / after they are reset.")
                return None
            else:
                batch_size, input_length, _ = value.shape
        keys.append(key)
    return layers, keys, input_length, batch_size


def read_hidden_states(probe_hidden_output):
    if not isinstance(probe_hidden_output, list):
         probe_hidden_output = [probe_hidden_output]
    n_tokens_generated = len(probe_hidden_output)
    if n_tokens_generated == 0:
        warnings.warn("No tokens generated yet or already reset")
        return None

    details = detect_hidden_states_parameters(probe_hidden_output[0])
    if details is None:
        return None
    layers, keys, input_length, batch_size = details
    ret = {}
    for layer in layers:
        ret[f"layer_{layer}"] = {}
        for key in keys:
            ret[f"layer_{layer}"][key] = [probe_hidden_output[0][layer][key]]
    for hidden_output in probe_hidden_output[1:]:
         for layer in layers:
              for key in keys:
                   ret[f"layer_{layer}"][key].append(hidden_output[layer][key])
    for layer in layers:
         for key in keys:
              ret[f"layer_{layer}"][key] = torch.cat(ret[f"layer_{layer}"][key], dim=1)[0] # batch size is always 0
    return ret     

def to_numpy(hidden_states):
     for layer in hidden_states.keys():
          for key in hidden_states[layer].keys():
               hidden_states[layer][key] = hidden_states[layer][key].numpy()

def to_numpy_list(hidden_states_list):
    for i in range(len(hidden_states_list)):
        to_numpy(hidden_states_list[i])


name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token = tokenizer.eos_token
config = AutoConfig.from_pretrained(name)
set_tracking_config(config, track_layers=[2, 15, 30], track_mlp=True, track_attention=False)

model = AutoModelForCausalLM.from_pretrained(name, config=config)
model.config.pad_token_id = model.config.eos_token_id

prompts = ["What was Einsteins first name? ", "What is the capital of Paris? "]
inputs = tokenizer(prompts, return_tensors="pt").to(model.device)
hidden_states_list = []
for i in tqdm(range(5)): 
    output = model.generate(**inputs, max_new_tokens=3)
    hidden_states = read_hidden_states(model.probe_hidden_output)
    hidden_states_list.append(hidden_states)
    model.probe_reset_hidden_output()

data_pkl = "hidden_states_torch.pkl"
with open(data_pkl, "wb") as f:
    pickle.dump(hidden_states_list, f)
to_numpy_list(hidden_states_list)
data_np_pkl = "hidden_states_np.pkl"
with open(data_np_pkl, "wb") as f:
    pickle.dump(hidden_states_list, f)
data_npz = "hidden_states.npz"
np.savez(data_npz, hidden_states_list)
print("Saved to all formats")


