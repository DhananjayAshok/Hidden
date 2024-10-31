import click
import pandas as pd
import numpy as np
from compute_hidden import alt_load_hidden_states
import os
import warnings
from sklearn.linear_model import LogisticRegression
from metrics import get_best_accuracy

label_map = {"unanswerable": "unanswerable", "confidence": "correct"}
results_dir = os.getenv("RESULTS_DIR")
data_dir = os.getenv("DATA_DIR")


@click.command()
@click.option("--task", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--prediction_dir", type=str, default=None)
@click.option('--random_sample_train', type=int, default=None)
@click.option('--random_sample_test', type=int, default=None)
@click.option('--random_seed', type=int, default=42)
@click.option('--model_kind', type=click.Choice(['linear', 'mlp', 'transformer'], case_sensitive=False), default="linear")
def main(task, dataset, prediction_dir, random_sample_train, random_sample_test, random_seed, model_kind):
    np.random.seed(random_seed)
    if prediction_dir is not None:
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
    hidden_states_dir = f"{results_dir}/{task}/"
    train_df = pd.read_csv(f"{results_dir}/{task}/{dataset}_train_inference.csv")
    test_df = pd.read_csv(f"{results_dir}/{task}/{dataset}_test_inference.csv")
    train_indices = None
    test_indices = None
    train_start_idx = 0
    train_end_idx = None
    test_start_idx = 0
    test_end_idx = None
    if random_sample_train is not None:
        train_df = train_df.sample(n=random_sample_train)
        train_indices = list(train_df.index)  # should make this idx
        train_start_idx = -2
        train_end_idx = -1
    if random_sample_test is not None:
        test_df = test_df.sample(n=random_sample_test)
        test_indices = list(test_df.index)
        test_start_idx = -2
        test_end_idx = -1
    X_train, train_keep_indices = alt_load_hidden_states(f"{hidden_states_dir}/train/{dataset}/", start_idx=train_start_idx, end_idx=train_end_idx, include_files=train_indices)
    X_test, test_keep_indices = alt_load_hidden_states(f"{hidden_states_dir}/test/{dataset}/", start_idx=test_start_idx, end_idx=test_end_idx, include_files=test_indices)
    train_df = train_df.iloc[train_keep_indices].reset_index(drop=True)
    test_df = test_df.iloc[test_keep_indices].reset_index(drop=True)
    y_train = train_df[label_map[task]].values
    y_test = test_df[label_map[task]].values
    if model_kind == "linear":
        model = Linear()
    else:
        raise ValueError(f"Model kind {model_kind} not implemented")
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)
    test_pred = model.predict_proba(X_test)
    train_score = get_best_accuracy(y_train, train_pred)
    test_score = get_best_accuracy(y_test, test_pred)
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
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
if __name__ == "__main__":
    main()