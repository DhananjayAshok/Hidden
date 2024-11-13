import click
import pandas as pd
import numpy as np
from compute_hidden import alt_load_hidden_states
import os
import warnings
import pickle
from models import get_model
from metrics import compute_accuracies

offset_map = {"unanswerable": 0, "confidence": 5}
results_dir = os.getenv("RESULTS_DIR")
data_dir = os.getenv("DATA_DIR")


def get_xydf(task, dataset, model_save_name, split="train", random_sample=None, task_offset=0, random_seed=42):
    assert split in ["train", "test"]
    hidden_states_dir = f"{results_dir}/{model_save_name}/{task}/"
    try:
        df = pd.read_csv(f"{results_dir}/{model_save_name}/{task}/{dataset}_{split}_inference.csv")
    except:
        if "indosentiment" in dataset:
            df = pd.read_csv(f"{results_dir}/{model_save_name}/{task}/{dataset}_{split}_inference.csv", lineterminator="\n")
        else:
            raise ValueError(f"Could not find {dataset}_{split}_inference.csv in {results_dir}/{model_save_name}/{task}/")
    if "label" not in df.columns:
        raise ValueError(f"Label column not found in {dataset}_{split}_inference.csv with columns {df.columns}. You probably need to run either evaluate.py or metrics.py to add a label column. If thats not the issue rip bro. ")
    available_indices = [int(f) for f in os.listdir(f"{hidden_states_dir}/{split}/{dataset}/")]
    df = df[df.index.isin(available_indices)]
    indices = None
    start_idx = 0
    end_idx = None
    if random_sample is not None:
        random_sample = min(random_sample, len(df))
    else:
        random_sample = len(df)
    df = df.sample(n=random_sample, random_state=random_seed)
    indices = list(df.index)
    start_idx = -2
    end_idx = -1
    X, keep_indices = alt_load_hidden_states(f"{hidden_states_dir}/{split}/{dataset}/", start_idx=start_idx, end_idx=end_idx, include_files=indices, task_offset=task_offset)
    df = df.loc[keep_indices].reset_index(drop=True)
    y = df["label"].values
    return X, y, df


def do_model_fit(model, X_train, y_train, X_test, y_test, verbose=True):
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)
    test_pred = model.predict_proba(X_test)
    train_scores, train_thresholds = compute_accuracies(y_train, train_pred)
    test_scores, test_thresholds = compute_accuracies(y_test, test_pred)
    #for i in range(len(train_scores)):
    #    train_scores[i] = int(train_scores[i] * 10000) / 100
    #    test_scores[i] = int(test_scores[i] * 10000) / 100
    #for i in range(0, len(train_thresholds), 10):
    #    print(f"Threshold: {train_thresholds[i]}| Train Accuracy: {train_scores[i]}| Test Accuracy: {test_scores[i]}")
    around_50 = np.argmin(np.abs(np.array(train_thresholds) - 0.5))
    test_acc = test_scores[around_50]
    if verbose:
        print(f"Base rate: {y_train.mean()} (Train), {y_test.mean()} (Test)")
        print(f"Final Test Accuracy: {test_acc}")
    return train_pred, test_pred, test_acc

@click.command()
@click.option("--task", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--model_save_name", type=str, default=None)
@click.option("--prediction_dir", type=str, default=None)
@click.option('--random_sample_train', type=int, default=None)
@click.option('--random_sample_test', type=int, default=None)
@click.option('--random_seed', type=int, default=42)
@click.option('--model_kind', type=click.Choice(['linear', 'mlp', 'transformer'], case_sensitive=False), default="linear")
def main(task, dataset, model_save_name, prediction_dir, random_sample_train, random_sample_test, random_seed, model_kind):
    np.random.seed(random_seed)
    if prediction_dir is not None:
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
    task_offset = offset_map[task] if False else 0
    X_train, y_train, train_df = get_xydf(task, dataset, model_save_name, "train", random_sample_train, task_offset=task_offset, random_seed=random_seed)
    X_test, y_test, test_df = get_xydf(task, dataset, model_save_name, "test", random_sample_test, task_offset=task_offset, random_seed=random_seed)
    model = get_model(model_kind)
    train_pred, test_pred, test_accuracy = do_model_fit(model, X_train, y_train, X_test, y_test)
    #print(f"Final Test Accuracy: {test_accuracy}")
    if prediction_dir is not None:
        train_df["probe_prediction"] = train_pred
        test_df["probe_prediction"] = test_pred
        train_df.to_csv(f"{prediction_dir}/train_pred.csv", index=False)
        test_df.to_csv(f"{prediction_dir}/test_pred.csv", index=False)
        model.save(f"{prediction_dir}/")
    return
    
if __name__ == "__main__":
    main()