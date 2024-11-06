import pandas as pd
import os
import click 
from prompt_engines import get_prompt_engine

@click.command()
@click.option("--task", type=str, required=True)
@click.option('--dataset', type=str, required=True)
@click.option("--method_name", type=str, required=True)
@click.option('--keep_strategy', type=click.Choice(['all', '0', '1', 'none'], case_sensitive=False), default="all")
@click.option('--text_column', type=str, default="text")
@click.option('--label_column', type=str, default="label")
@click.option('--sample_size', type=int, default=None)
@click.option('--random_seed', type=int, default=42)
def main(task, dataset, method_name, keep_strategy, text_column, label_column, sample_size, random_seed):
    if label_column == "none":
        label_column = None
    data_dir = os.getenv('DATA_DIR')
    data_file = f"{data_dir}/{task}/{dataset}_train.csv"
    df = pd.read_csv(data_file)
    max_idx = df["idx"].max()
    if text_column not in df.columns:
        raise ValueError(f"Text column {text_column} not found in dataframe with columns {df.columns}")
    if label_column is not None and label_column not in df.columns:
        raise ValueError(f"Label column {label_column} not found in dataframe with columns {df.columns}")
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=random_seed)
    method = get_method(method_name)
    new_texts = []
    new_labels = []
    new_idxs = []
    running_idx = max_idx + 1
    for i in range(len(df)):
        text = df.loc[i, text_column]
        label = df.loc[i, label_column] if label_column is not None else None
        new_text, new_label = method(text, label)
        new_texts.append(new_text)
        new_labels.append(new_label)
        new_idxs.append(running_idx)
        running_idx += 1
    new_df = pd.DataFrame({text_column: new_texts, label_column: new_labels, "idx": new_idxs})
    if keep_strategy == "all":
        df = pd.concat([df, new_df], axis=0, ignore_index=True)
    elif keep_strategy == "0":
        df = df[df[label_column] == 0]
        df = pd.concat([df, new_df], axis=0, ignore_index=True).reset_index(drop=True)
    elif keep_strategy == "1":
        df = df[df[label_column] == 1]
        df = pd.concat([df, new_df], axis=0, ignore_index=True).reset_index(drop=True)
    elif keep_strategy == "none":
        df = new_df
    save_file = f"{data_dir}/{task}/{dataset}_synth_train.csv"
    df.to_csv(save_file, index=False)


def get_method(method_name):
    if "-" in method_name:
        prompt_engine = method_name.split("-")[1]
        method_name = method_name.split("-")[0]
    if method_name == "nei":
        return NEIGenerator(prompt_engine)
    raise ValueError(f"Method {method_name} not found")


class SyntheticGenerator:
    def __call__(self, text, label):
        prompt, target_label = self.get_prompt(text, label)
        return self.parse_fn(self.prompt_engine()), target_label

    @staticmethod
    def parse_normal(text):
        return text
    
    @staticmethod
    def parse_cot(text):
        if "||" in text:
            return text.split("||")[-1]
        return text

    def get_prompt(self, text):
        raise NotImplementedError

class NEIGenerator:
    def __init__(self, prompt_engine_name):
        self.prompt_engine = get_prompt_engine(prompt_engine_name)
        self.parse_fn = self.parse_normal


    def get_prompt(self, text, label):
        evidence = None
        claim = None
        prompt = f"Given the claim: '{claim}' and the following evidence `{evidence}', rewrite the evidence such that there is not enough information in the evidence to confirm or deny the claim: \nRewritten Evidence: "
        return prompt, 1

if __name__ == "__main__":
    main()