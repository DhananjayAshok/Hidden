import click
import pandas as pd
import numpy as np
from compute_hidden import alt_load_hidden_states
import os
import warnings
import pickle
from models import get_model
from metrics import compute_accuracies

label_map = {"unanswerable": "unanswerable", "confidence": "correct"}
offset_map = {"unanswerable": 0, "confidence": 5}
results_dir = os.getenv("RESULTS_DIR")
data_dir = os.getenv("DATA_DIR")


def get_xydf(task, dataset, split="train", random_sample=None, exclude_layers=[], exclude_hidden=[], task_offset=0):
    assert split in ["train", "test"]
    hidden_states_dir = f"{results_dir}/{task}/"
    df = pd.read_csv(f"{results_dir}/{task}/{dataset}_{split}_inference.csv")
    available_indices = [int(f) for f in os.listdir(f"{hidden_states_dir}/{split}/{dataset}/")]
    df = df[df.index.isin(available_indices)]
    indices = None
    start_idx = 0
    end_idx = None
    if random_sample is not None:
        random_sample = min(random_sample, len(df))
    else:
        random_sample = len(df)
    df = df.sample(n=random_sample)
    indices = list(df.index)
    start_idx = -2
    end_idx = -1
    X, keep_indices = alt_load_hidden_states(f"{hidden_states_dir}/{split}/{dataset}/", start_idx=start_idx, end_idx=end_idx, include_files=indices, exclude_files=exclude_hidden, exclude_layers=exclude_layers, task_offset=task_offset)
    df = df.loc[keep_indices].reset_index(drop=True)
    y = df[label_map[task]].values
    return X, y, df


def do_model_fit(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)
    test_pred = model.predict_proba(X_test)
    train_scores, train_thresholds = compute_accuracies(y_train, train_pred)
    test_scores, test_thresholds = compute_accuracies(y_test, test_pred)
    for i in range(len(train_scores)):
        train_scores[i] = int(train_scores[i] * 10000) / 100
        test_scores[i] = int(test_scores[i] * 10000) / 100
    print(f"Base rate: {y_train.mean()} (Train), {y_test.mean()} (Test)")
    for i in range(0, len(train_thresholds), 10):
        print(f"Threshold: {train_thresholds[i]}| Train Accuracy: {train_scores[i]}| Test Accuracy: {test_scores[i]}")
    return train_pred, test_pred

@click.command()
@click.option("--task", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--prediction_dir", type=str, default=None)
@click.option('--random_sample_train', type=int, default=None)
@click.option('--random_sample_test', type=int, default=None)
@click.option('--random_seed', type=int, default=42)
@click.option('--model_kind', type=click.Choice(['linear', 'mlp', 'transformer'], case_sensitive=False), default="linear")
@click.option("--exclude_layers", multiple=True, default=[])
@click.option("--exclude_hidden", multiple=True, default=[])
@click.option('use_task_offset', type=bool, default=False)
def main(task, dataset, prediction_dir, random_sample_train, random_sample_test, random_seed, model_kind, exclude_layers, exclude_hidden, use_task_offset):
    np.random.seed(random_seed)
    if prediction_dir is not None:
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
    task_offset = offset_map[task] if use_task_offset else 0
    X_train, y_train, train_df = get_xydf(task, dataset, "train", random_sample_train, exclude_layers, exclude_hidden, task_offset=task_offset)
    X_test, y_test, test_df = get_xydf(task, dataset, "test", random_sample_test, exclude_layers, exclude_hidden, task_offset=task_offset)
    model = get_model(model_kind)
    train_pred, test_pred = do_model_fit(model, X_train, y_train, X_test, y_test)
    if prediction_dir is not None:
        train_df["probe_prediction"] = train_pred
        test_df["probe_prediction"] = test_pred
        train_df.to_csv(f"{prediction_dir}/train_pred.csv", index=False)
        test_df.to_csv(f"{prediction_dir}/test_pred.csv", index=False)
        model.save(f"{prediction_dir}/")
    return
    
if __name__ == "__main__":
    main()