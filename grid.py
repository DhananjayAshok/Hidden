import click
import pandas as pd
import numpy as np
from compute_hidden import alt_load_hidden_states, null_state_processor, numpy_state_processor
import os
import warnings
import pickle
from models import get_model_suite
from iid_modeling import do_model_fit, offset_map
from tqdm import tqdm
import json


results_dir = os.getenv("RESULTS_DIR")
data_dir = os.getenv("DATA_DIR")


def get_xyall(task, dataset, model_save_name, random_sample=None, random_seed=42):
    split = "train"
    hidden_states_dir = f"{results_dir}/{model_save_name}/{task}/"
    df = pd.read_csv(f"{results_dir}/{model_save_name}/{task}/{dataset}_{split}_inference.csv")
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
    X, keep_indices = alt_load_hidden_states(f"{hidden_states_dir}/{split}/{dataset}/", start_idx=start_idx, end_idx=end_idx, include_files=indices, state_processor=null_state_processor)
    df = df.loc[keep_indices].reset_index(drop=True)
    y = df["label"].values
    return X, y, df


def get_config_X(X, config, task):
    task_offset = offset_map[task] if config['use_task_offset'] else 0
    # remove task_offset from config
    new_config = config.copy()
    new_config.pop('use_task_offset', None)
    X_all = np.array([numpy_state_processor(x, task_offset=task_offset, **new_config) for x in X])
    return X_all


def do_fold_fit(X_all, y_all, model, n_fold):
    # Shuffle X_all and y_all
    indices = np.arange(len(X_all))
    np.random.shuffle(indices)
    X_all = X_all[indices]
    y_all = y_all[indices]
    fold_size = len(X_all) // n_fold
    fold_accs = []
    for i in range(n_fold):
        X_train = np.concatenate([X_all[:i * fold_size], X_all[(i + 1) * fold_size:]], axis=0)
        y_train = np.concatenate([y_all[:i * fold_size], y_all[(i + 1) * fold_size:]], axis=0)
        X_test = X_all[i * fold_size:(i + 1) * fold_size]
        y_test = y_all[i * fold_size:(i + 1) * fold_size]
        train_pred, test_pred, test_accuracy = do_model_fit(model, X_train, y_train, X_test, y_test, verbose=False)
        fold_accs.append(test_accuracy)
    arr = np.array(fold_accs)
    return arr.mean(), arr.std()



@click.command()
@click.option("--task", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--model_save_name", type=str, default=None)
@click.option('--n_samples', type=int, default=None)
@click.option('--n_fold', type=int, default=5)
@click.option('--random_seed', type=int, default=42)
@click.option('--report_path', type=str, default="reports/")
@click.option('--config_path', type=str, default="configs/base.json")
@click.option('--suite_name', type=click.Choice(['base', 'linear', 'tree'], case_sensitive=False), default="base")
def main(task, dataset, model_save_name, n_samples, n_fold, random_seed, report_path, config_path, suite_name):
    if not os.path.exists(report_path):
        os.makedirs(report_path)
    if os.path.exists(f"{report_path}/{task}_{dataset}_{random_seed}.csv"):
        print(f"Report already exists for {task} and {dataset} for random_seed {random_seed}. Skipping...")
        return
    np.random.seed(random_seed)    
    columns = ["config", "model", "acc_mean", "acc_std"]
    data = []
    with open(config_path, "r") as f:
        configs = json.load(f)
    X_pre, y_all, df = get_xyall(task, dataset, model_save_name, n_samples, random_seed)
    print(f"Base Rate: {y_all.mean()}")
    for config in tqdm(configs):
        print(f"Config: {config}")
        X_all = get_config_X(X_pre, config, task)
        model_suite = get_model_suite(suite_name)
        for model_name in model_suite:
            model = model_suite[model_name]
            acc_mean, acc_std = do_fold_fit(X_all, y_all, model, n_fold)
            print(f"\tModel: {model_name} | Mean: {acc_mean} | Std: {acc_std}")
            data.append([config, model_name, acc_mean, acc_std])
        del model_suite, X_all
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f"{report_path}/{task}_{dataset}_{random_seed}.csv", index=False)
    return
    
if __name__ == "__main__":
    main()