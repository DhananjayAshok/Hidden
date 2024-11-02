# Write tests for:
# Saving of hidden states
# Loading of hidden states
# Modeling of probes on trivial instance
# Modeling of probes on non-trivial instance
# -------------------------------------------
import unittest
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from compute_hidden import set_tracking_config

model_name = "meta-llama/Llama-3.1-8B-Instruct"

eps = 1e-5

def array_equal(a, b):
    return (a-b).abs().max() < eps

def compute_hidden(prompt, track_layers):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    set_tracking_config(config, track_layers=track_layers, track_mlp=True, track_attention=True, track_projection=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=5, stop_strings=["[STOP]"], pad_token_id=tokenizer.eos_token_id, tokenizer=tokenizer, output_hidden_states=True, output_attentions=True, return_dict_in_generate=True)
    breakpoint()
    return output

class TestHiddenStates(unittest.TestCase):
    def test_compute_hidden(self):
        pass


if __name__ == "__main__":
    compute_hidden()
