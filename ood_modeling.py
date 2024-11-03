import click
import pandas as pd
import numpy as np
from compute_hidden import alt_load_hidden_states
import os
import warnings
import pickle
from models import get_model
from iid_modeling import get_xydf, do_model_fit

task_datasets = {"unanswerable": ["qnota", "selfaware", "known_known"], "toxicity_avoidance": ["real_toxicity_prompts", "toxic_chat"]}
results_dir = os.getenv("RESULTS_DIR")
data_dir = os.getenv("DATA_DIR")


@click.command()
@click.option("--task", type=str, required=True)
@click.option("--prediction_dir", type=str, default=None)
@click.option('--random_sample_train_per', type=int, default=None)
@click.option('--random_sample_test_per', type=int, default=None)
@click.option('--random_seed', type=int, default=42)
@click.option('--model_kind', type=click.Choice(['linear', 'mlp', 'transformer'], case_sensitive=False), default="linear")
def main(task, dataset, prediction_dir, random_sample_train_per, random_sample_test_per, random_seed, model_kind):
    np.random.seed(random_seed)
    if prediction_dir is not None:
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
    
    data = {}
    for dataset in task_datasets[task]:
        X_train, y_train, train_df = get_xydf(task, dataset, "train", random_sample_train_per, random_seed=random_seed)
        X_test, y_test, test_df = get_xydf(task, dataset, "test", random_sample_test_per, random_seed=random_seed)
        internal_data = {"X_train": X_train, "y_train": y_train, "train_df": train_df, "X_test": X_test, "y_test": y_test, "test_df": test_df}
        data[dataset] = internal_data        
    
    print(f"Datasets Available for {task} task: {task_datasets[task]}")
    for dataset in data:
        print(f"Fitting model for test: {dataset}")
        model = get_model(model_kind)
        train_pred, test_pred, train_df, test_df = fit_one_set(model, data, dataset)
        del model
    return
    
def fit_one_set(model, data, test_dataset): # TODO: Must debug this
    train_datasets = [dataset for dataset in data.keys() if dataset != test_dataset]
    X_train = np.concatenate([data[dataset]["X_train"] for dataset in train_datasets], axis=0)
    y_train = np.concatenate([data[dataset]["y_train"] for dataset in train_datasets], axis=0)
    train_df = pd.concat([data[dataset]["train_df"] for dataset in train_datasets], axis=0, ignore_index=True)
    X_test = data[test_dataset]["X_test"]
    y_test = data[test_dataset]["y_test"]
    test_df = data[test_dataset]["test_df"]
    train_pred, test_pred = do_model_fit(model, X_train, y_train, X_test, y_test)
    return train_pred, test_pred, train_df, test_df
    

if __name__ == "__main__":
    main()