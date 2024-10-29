import click
import pandas as pd
from tqdm import tqdm
import warnings
import os

@click.command()
@click.option("--file", type=str, required=True)
@click.option("--pred_column", type=str, default="output")
@click.option("--label_column", type=str, default="label")
@click.option('--correct_column', type=str, default="correct")
@click.option('--overwrite', type=bool, default=False)
@click.option("--cot", type=bool, default=False)
def main(file, pred_column, label_column, correct_column, overwrite, cot):
    df = pd.read_csv(file)
    if pred_column not in df.columns:
        raise ValueError(f"Prediction column {pred_column} not found in dataframe")
    if label_column not in df.columns:
        raise ValueError(f"Label column {label_column} not found in dataframe")
    if correct_column in df.columns and not overwrite:
        raise ValueError(f"Correct column {correct_column} already found in dataframe. Set overwrite to True to overwrite")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row.isna()[pred_column] or row.isna()[label_column]:
            continue
        prediction = row[pred_column]
        label = row[label_column]
        if cot:
            correct = cot_check(prediction, label)
        else:
            correct = normal_check(prediction, label)
        df.loc[i, correct_column] = correct
    df.to_csv(file, index=False)

def cot_check(prediction, label):
    pred_labels = prediction.split("||")
    if len(pred_labels) == 1:
        warnings.warn(f"Supposed to be COT but prediction {prediction} does not have ||")
        pred_labels = pred_labels[0]
    elif len(pred_labels) == 2:
        pred_labels = pred_labels[1]
    else:
        warnings.warn(f"Supposed to be COT but prediction {prediction} has more than 1 ||")
        pred_labels = pred_labels[1]
    return normal_check(pred_labels, label)

def normal_check(prediction, label):
    return prediction.lower().strip() == label.lower().strip()
