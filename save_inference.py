import click
from compute_hidden import set_tracking_config, read_hidden_states, save_hidden_states, alt_save_hidden_states
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import os


@click.command()
@click.option("--model_name", type=str, required=True)
@click.option("--data_path", type=str, required=True)
@click.option("--output_csv_path", type=str, required=True)
@click.option("--output_hidden_dir", type=str, required=True)
@click.option("--save_every", type=int, default=500)
@click.option("--start_idx", type=int, default=0)
@click.option('--stop_idx', type=int, default=None)
@click.option("--max_new_tokens", type=int, default=10)
@click.option("--stop_strings", type=str, default="[STOP]")
@click.option("--remove_stop_strings", type=bool, default=True)
@click.option("--track_layers", type=str, default="2,15,30")
@click.option("--track_mlp", type=bool, default=True)
@click.option("--track_attention", type=bool, default=True)
@click.option("--track_projection", type=bool, default=True)
def main(model_name, data_path, output_csv_path, output_hidden_dir, save_every, start_idx, stop_idx, max_new_tokens, stop_strings, remove_stop_strings, track_layers, track_mlp, track_attention, track_projection):
    track_layers = [int(x) for x in track_layers.split(",")]
    stop_strings = stop_strings.split(",")
    makedirs = [output_hidden_dir]
    for output_path in [output_csv_path]:
        if os.path.dirname(output_path) != "":
            makedirs.append(os.path.dirname(output_path))
    for directory in makedirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    set_tracking_config(config, track_layers=track_layers, track_mlp=track_mlp, track_attention=track_attention, track_projection=track_projection)
    data_df = pd.read_csv(data_path)
    assert "text" in data_df.columns
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    hidden_states_list = []
    if stop_idx is not None:
        stop_idx = min(stop_idx, len(data_df))
    else:
        stop_idx = len(data_df)
    for i in tqdm(range(start_idx, stop_idx)):
        prompt = data_df.loc[i, "text"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, stop_strings=stop_strings, pad_token_id=tokenizer.eos_token_id, tokenizer=tokenizer)
        hidden_states = read_hidden_states(model.probe_hidden_output)
        model.probe_reset_hidden_output()
        hidden_states_list.append(hidden_states)
        # currently hardcoding in layer_2 attention
        assert hidden_states_list[-1]["layer_2"]["attention"].shape[0] <= output.shape[1] - 1
        assert output.shape[1] <= input_length + max_new_tokens
        output_only = output[0, input_length:]
        out = tokenizer.decode(output_only, skip_special_tokens=True)
        if remove_stop_strings:
            for stop_string in stop_strings:
                out = out.replace(stop_string, "")
        data_df.loc[i, "output"] = out
        if (i % save_every == 0 and i > 0) or i == stop_idx - 1:
            alt_save_hidden_states(hidden_states_list, output_hidden_dir, i, exists_ok=True)
            hidden_states_list = []
            if "label" in data_df.columns:
                data_df[["output", "label"]].to_csv(output_csv_path, index=False)
            else:
                data_df[["output"]].to_csv(output_csv_path, index=False)
    return 

if __name__ == "__main__":
    main()