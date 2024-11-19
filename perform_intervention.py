import click
from compute_hidden import set_tracking_config, read_hidden_states, save_hidden_states, alt_save_hidden_states
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


@click.command()
@click.option("--model_name", type=str, required=True)
@click.option("--data_path", type=str, required=True)
@click.option("--intervention_vector_path", type=str, required=True)
@click.option('--intervention_layer', type=int, required=True)
@click.option('--intervention_strength', type=float, required=True, multiple=True)
@click.option("--intervention_location", type=click.Choice(["mlp", "attention"], case_sensitive=False), required=True)
@click.option("--output_csv_path", type=str, required=True)
@click.option("--start_idx", type=int, default=0)
@click.option('--stop_idx', type=int, default=None)
@click.option("--max_new_tokens", type=int, default=10)
@click.option("--stop_strings", type=str, default="[STOP]")
@click.option("--remove_stop_strings", type=bool, default=True)
@click.option("--save_every", type=int, default=100)
def main(model_name, data_path, intervention_vector_path, intervention_layer, intervention_strength, intervention_location, output_csv_path, start_idx, stop_idx, max_new_tokens, stop_strings, remove_stop_strings, save_every):
    stop_strings = stop_strings.split(",")
    if os.path.dirname(output_csv_path) != "":
        if not os.path.exists(os.path.dirname(output_csv_path)):
            os.makedirs(os.path.dirname(output_csv_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    shift_activation = {intervention_layer: None}
    array = np.load(intervention_vector_path)
    if array.shape[-1] != config.hidden_size:
        raise ValueError(f"Intervention vector shape {array.shape} does not match hidden size {config.hidden_size}")
    try:
        array = array.reshape(config.hidden_size)
    except:
        raise ValueError(f"Intervention vector shape {array.shape} does not match hidden size {config.hidden_size}")
    shift_activation[intervention_layer] = {intervention_location: {"vector": array, "strength": intervention_strength[0]}}
    set_tracking_config(config, shift_activations=shift_activation)
    try:
        data_df = pd.read_csv(data_path)
    except:
        assert "indosentiment" in data_path
        data_df = pd.read_csv(data_path, lineterminator="\n")
    assert "text" in data_df.columns or "prompt" in data_df.columns
    if "prompt" in data_df.columns:
        text_col = "prompt"
    else:
        text_col = "text"
    for i in range(len(intervention_strength)):
        data_df[f"output_{intervention_strength}"] = None
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    if stop_idx is not None:
        stop_idx = min(stop_idx, len(data_df))
    else:
        stop_idx = len(data_df)
    for i in tqdm(range(start_idx, stop_idx)):
        prompt = data_df.loc[i, text_col]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]
        for j, strength in enumerate(intervention_strength):
            model.set_intervention_strength(strength)
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, stop_strings=stop_strings, pad_token_id=tokenizer.eos_token_id, tokenizer=tokenizer)
            #model.probe_reset_hidden_output() should not be needed
            output_only = output[0, input_length:]
            out = tokenizer.decode(output_only, skip_special_tokens=True)
            if remove_stop_strings:
                for stop_string in stop_strings:
                    out = out.replace(stop_string, "")
            data_df.loc[i, f"output_{strength}"] = out
        if (i % save_every == 0 and i > 0) or i == stop_idx - 1:
            data_df.to_csv(output_csv_path, index=False)
    return 

if __name__ == "__main__":
    main()