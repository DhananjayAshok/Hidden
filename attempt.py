from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import numpy as np
import torch

name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

prompt = "What was Einsteins first name? "


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
        ret[layer] = {}
        for key in keys:
            ret[layer][key] = [probe_hidden_output[0][layer][key]]
    for hidden_output in probe_hidden_output[1:]:
         for layer in layers:
              for key in keys:
                   ret[layer][key].append(hidden_output[layer][key])
    for layer in layer:
         for key in key:
              ret[layer][key] = torch.cat(ret[layer][key], dim=1)
    return ret



inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=10)
hidden_states = read_hidden_states(model.probe_hidden_output)
print(tokenizer.decode(output[0], skip_special_tokens=True))

