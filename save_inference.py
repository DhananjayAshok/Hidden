import click
from compute_hidden import set_tracking_config, read_hidden_states, save_hidden_states
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import os


@click.command()
@click.option("--model_name", type=str, required=True)
@click.option("--data_path", type=str, required=True)
@click.option("--output_csv_path", type=str, required=True)
@click.option("--output_hidden_path", type=str, required=True)
@click.option("--max_new_tokens", type=int, default=10)
@click.option("--stop_string", type=str, default="[STOP]")
@click.option("--remove_stop_string", type=bool, default=True)
@click.option("--track_layers", type=str, default="2,15,30")
@click.option("--track_mlp", type=bool, default=True)
@click.option("--track_attention", type=bool, default=True)
def main(model_name, data_path, output_csv_path, output_hidden_path, max_new_tokens, stop_string, remove_stop_string, track_layers, track_mlp, track_attention):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    set_tracking_config(config, track_layers=track_layers, track_mlp=track_mlp, track_attention=track_attention)
    data_df = pd.read_csv(data_path)
    assert "text" in data_df.columns
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    hidden_states_list = []
    for i in tqdm(range(len(data_df))):
        prompt = data_df.loc[i, "text"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, stop_strings=[stop_string], pad_token_id=tokenizer.eos_token_id)
        hidden_states = read_hidden_states(model.probe_hidden_output)
        model.probe_reset_hidden_output()
        hidden_states_list.append(hidden_states)
        output_only = output[0, input_length:]
        out = tokenizer.decode(output_only, skip_special_tokens=True)
        if remove_stop_string:
            out = out.replace(stop_string, "")
        data_df.loc[i, "output"] = out
    for output_path in [output_csv_path, output_hidden_path]:
        if os.path.dirname(output_path) != "":
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
    save_hidden_states(hidden_states_list, output_hidden_path)
    if "label" in data_df.columns:
        data_df[["output", "label"]].to_csv(output_csv_path, index=False)
    else:
        data_df[["output"]].to_csv(output_csv_path, index=False)
    return 