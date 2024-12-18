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


def get_xydf(task, dataset, model_save_name, split="train", random_sample=None, task_offset=0, random_seed=42, only_mlp=False, only_attention=False, only_layer=None):
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
    exclude_layers = []
    exclude_hidden = []
    if only_mlp:
        exclude_hidden = ["attention", "projection"]
    if only_attention:
        exclude_hidden = ["projection", "mlp"]
    if only_layer is not None:
        exclude_layers = [f"layer_{i}" for i in range(100) if i != only_layer]

    X, keep_indices = alt_load_hidden_states(f"{hidden_states_dir}/{split}/{dataset}/", start_idx=start_idx, end_idx=end_idx, include_files=indices, task_offset=task_offset, exclude_layers=exclude_layers, exclude_hidden=exclude_hidden)
    df = df.loc[keep_indices].reset_index(drop=True)
    if df["label"].isnull().sum() > 0:
        raise ValueError(f"Found {df['label'].isnull().sum()} null labels in {dataset}_{split}_inference.csv")
    if df["label"].nunique() != 2:
        warnings.warn(f"Found {df['label'].nunique()} unique labels in {dataset}_{split}_inference.csv. This is not a binary classification task.")
    df["label"] = df["label"].astype(int)
    y = df["label"].values
    return X, y, df


def do_model_fit(model, X_train, y_train, X_test, y_test, train_df, test_df, verbose=True, prediction_dir=None):
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
    if prediction_dir is not None:
        train_df["probe_prediction"] = train_pred
        test_df["probe_prediction"] = test_pred
        train_df.to_csv(f"{prediction_dir}/train_pred.csv", index=False)
        test_df.to_csv(f"{prediction_dir}/test_pred.csv", index=False)
        model.save(f"{prediction_dir}/")
    return train_pred, test_pred, test_acc

@click.command()
@click.option("--task", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--model_save_name", type=str, required=True)
@click.option('--prediction_dir', type=str, default=None)
@click.option('--random_sample_train', type=int, default=None)
@click.option('--random_sample_test', type=int, default=None)
@click.option('--random_seed', type=int, default=42)
@click.option('--model_kind', type=click.Choice(['linear', 'mean', 'mlp', 'transformer'], case_sensitive=False), default="linear")
@click.option('--only_mlp', type=bool, default=False)
@click.option('--only_attention', type=bool, default=False)
@click.option('--only_layer', type=int, default=None)
def main(task, dataset, model_save_name, prediction_dir, random_sample_train, random_sample_test, random_seed, model_kind,  only_mlp, only_attention, only_layer):
    assert not (only_mlp and only_attention), "Cannot specify both only_mlp and only_attention"
    np.random.seed(random_seed)
    if prediction_dir is not None:
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
    task_offset = offset_map[task] if False else 0
    X_train, y_train, train_df = get_xydf(task, dataset, model_save_name, "train", random_sample_train, task_offset=task_offset, random_seed=random_seed, only_mlp=only_mlp, only_attention=only_attention, only_layer=only_layer)
    X_test, y_test, test_df = get_xydf(task, dataset, model_save_name, "test", random_sample_test, task_offset=task_offset, random_seed=random_seed, only_mlp=only_mlp, only_attention=only_attention, only_layer=only_layer)
    model = get_model(model_kind)
    train_pred, test_pred, test_accuracy = do_model_fit(model, X_train, y_train, X_test, y_test, train_df, test_df, verbose=True, prediction_dir=prediction_dir)
    return
    
if __name__ == "__main__":
    main()