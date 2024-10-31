from transformers import pipeline
import click
import pandas as pd
from tqdm import tqdm
import os

@click.command()
@click.option("--file", type=str, required=True)
@click.option("--metric_name", type=str, required=True)
@click.option("--input_column", type=str, default="text")
@click.option("--output_column", type=str, default="label")
@click.option("--overwrite", type=bool, default=False)
def main(file, metric_name, input_column, output_column, overwrite):
    df = pd.read_csv(file)
    if input_column not in df.columns:
        raise ValueError(f"Input column {input_column} not found in dataframe")
    if output_column in df.columns and not overwrite:
        raise ValueError(f"Output column {output_column} already found in dataframe. Set overwrite to True to overwrite")
    metric = get_metric_class(metric_name)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row.isna()[input_column]:
            continue
        input_text = row[input_column]
        output = metric(input_text)
        df.loc[i, output_column] = output
    df.to_csv(file, index=False)


def get_metric_class(metric_name):
    """
    Returns a class that can be called with input text and outputs a single float value
    """
    if metric_name == "toxdect-roberta":
        return ToxDectRoberta()


class ToxDectRoberta:
    def __init__(self):
        self.classifier = pipeline("text-classification",model='Xuhui/ToxDect-roberta-large', return_all_scores=True, device_map="auto")

    def __call__(self, text):
        output = self.classifier(text)
        scores = [0, 0]
        for scoredict in output[0]:
            label = scoredict["label"]
            score = scoredict["score"]
            if label == "LABEL_0":
                scores[0] = score
            elif label == "LABEL_1":
                scores[1] = score
        return scores[1]





if __name__ == "__main__":
    main()


