from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import warnings
import numpy as np
import torch
from tqdm import tqdm
import pickle
import h5py


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
              ret[f"layer_{layer}"][key] = torch.cat(ret[f"layer_{layer}"][key], dim=1)[0].numpy() # batch size is always 0
    return ret     

def to_numpy(hidden_states):
     for layer in hidden_states.keys():
          for key in hidden_states[layer].keys():
               hidden_states[layer][key] = hidden_states[layer][key].numpy()

def to_numpy_list(hidden_states_list):
    for i in range(len(hidden_states_list)):
        to_numpy(hidden_states_list[i])

def deepcopy(hidden_states):
    ret = {}
    for layer in hidden_states.keys():
        ret[layer] = {}
        for key in hidden_states[layer].keys():
            ret[layer][key] = hidden_states[layer][key].copy()
    return ret

name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token = tokenizer.eos_token
config = AutoConfig.from_pretrained(name)
set_tracking_config(config, track_layers=[2, 15, 30], track_mlp=True, track_attention=False)

model = AutoModelForCausalLM.from_pretrained(name, config=config)
model.config.pad_token_id = model.config.eos_token_id

prompt = "What was Einsteins first name? "
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=3)
hidden_states = read_hidden_states(model.probe_hidden_output)
model.probe_reset_hidden_output()
hidden_states_list = [hidden_states]
for i in tqdm(range(5)): 
    hidden_states = deepcopy(hidden_states)
    hidden_states_list.append(hidden_states)

with h5py.File('hidden_states.h5', 'w') as h5file:
    for idx, data in enumerate(hidden_states_list):
        group = h5file.create_group(f'item_{idx}')
        for layer_idx, layer_data in data.items():
            layer_grp = group.create_group(str(layer_idx))
            for key, array in layer_data.items():
                layer_grp.create_dataset(key, data=array)

data_np_pkl = "hidden_states.pkl"
with open(data_np_pkl, "wb") as f:
    pickle.dump(hidden_states_list, f)
data_npz = "hidden_states.npz"
np.savez(data_npz, hidden_states_list)
print("Saved to all formats")


