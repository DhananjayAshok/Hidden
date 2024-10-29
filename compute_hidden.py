import pickle
import torch
import warnings
import os
import numpy as np


def set_tracking_config(config, track_layers=[2, 15, 30], track_mlp=True, track_attention=True):
    config.track_layers = track_layers
    config.track_mlp = track_mlp
    config.track_attention = track_attention
    for element in track_layers:
         assert element < config.num_hidden_layers
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

def save_hidden_states(obj, filename):
     with open(filename, "wb") as f:
         pickle.dump(obj, f)

    
def load_hidden_states(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    

def alt_save_hidden_states(obj, folder, end_idx, exists_ok=False):
    os.makedirs(folder, exist_ok=exists_ok)
    for i in range(len(obj)):
        os.makedirs(f"{folder}/{end_idx - i}")
        layer_keys = list(obj[i].keys())
        for layer_key in layer_keys:
            os.makedirs(f"{folder}/{end_idx - i}/{layer_key}", exist_ok=exists_ok)
            hidden_keys = list(obj[i][layer_key].keys())
            for hidden_key in hidden_keys:
                item = obj[i][layer_key][hidden_key]
                filepath = f"{folder}/{end_idx - i}/{layer_key}/{hidden_key}.npy"
                np.save(filepath, item)
    return


def null_state_processor(x):
    return x


def numpy_state_processor(x):
    array = []
    layers = list(x.keys())
    layer_numbers = [int(layer.split("_")[1]) for layer in layers]
    layer_argsort = np.argsort(layer_numbers)
    layers = [layers[i] for i in layer_argsort]
    for layer_key in layers:
        hidden_keys = list(x[layer_key].keys())
        hidden_keys = sorted(hidden_keys)
        for hidden_key in hidden_keys:
            array.extend(x[layer_key][hidden_key][-1])
    return np.array(array)


def alt_load_hidden_states(filepath, start_idx, end_idx=None, state_processor=numpy_state_processor, ret_numpy=True):
    files = os.listdir(filepath)
    files = sorted([int(file) for file in files])
    if start_idx > files[-1]:
        warnings.warn("Start index is greater than the last file")
        return None
    elif end_idx is not None and end_idx < files[0]:
        warnings.warn("End index is less than the first file")
        return None
    if end_idx is not None:
        files = [file for file in files if file >= start_idx and file <= end_idx]
    else:
        files = [file for file in files if file >= start_idx]
    hidden_states_list = load_as_dict(files, filepath, state_processor)
    if ret_numpy:
        return np.array(hidden_states_list)
    else:
        return hidden_states_list


def load_as_dict(files, filepath, state_processor):
    hidden_states_list = []
    for file in files:
        layer_keys = os.listdir(f"{filepath}/{file}")
        hidden_states = {}
        for layer_key in layer_keys:
            hidden_states[layer_key] = {}
            hidden_keys = os.listdir(f"{filepath}/{file}/{layer_key}")
            for hidden_key in hidden_keys:
                item = np.load(f"{filepath}/{file}/{layer_key}/{hidden_key}")
                hidden_states[layer_key][hidden_key.split(".")[0]] = item
        hidden_states = state_processor(hidden_states)
        hidden_states_list.append(hidden_states)
    return hidden_states_list