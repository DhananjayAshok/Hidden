import click
import pandas as pd
import numpy as np
from compute_hidden import alt_load_hidden_states
import os
import warnings
from sklearn.linear_model import LogisticRegression

label_map = {"unanswerable": "unanswerable", "confidence": "correct"}
results_dir = os.getenv("RESULTS_DIR")
data_dir = os.getenv("DATA_DIR")


@click.command()
@click.option("--task", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option('model_kind', type=click.Choice(['linear', 'mlp', 'transformer'], case_sensitive=False), default="linear")
def main(task, dataset, model_kind):
    hidden_states_dir = f"{results_dir}/{task}/"
    X_train, train_keep_indices = alt_load_hidden_states(f"{hidden_states_dir}/train/{dataset}/")
    X_test, test_keep_indices = alt_load_hidden_states(f"{hidden_states_dir}/test/{dataset}/")
    train_df = pd.read_csv(f"{results_dir}/{task}/{dataset}_train_inference.csv")
    test_df = pd.read_csv(f"{results_dir}/{task}/{dataset}_test_inference.csv")
    breakpoint()
    train_df = train_df.iloc[train_keep_indices].reset_index(drop=True)
    test_df = test_df.iloc[test_keep_indices].reset_index(drop=True)
    y_train = train_df[label_map[task]].values
    y_test = test_df[label_map[task]].values
    if model_kind == "linear":
        model = Linear()
    else:
        raise ValueError(f"Model kind {model_kind} not implemented")
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Base rate: {y_train.mean()} (Train), {y_test.mean()} (Test)")
    print(f"Train score: {train_score}")
    print(f"Test score: {test_score}")
    return

class Linear:
    def __init__(self):
        self.model = LogisticRegression(random_state=0)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def score(self, X_train, y_train):
        return self.model.score(X_train, y_train)
    
if __name__ == "__main__":
    main()