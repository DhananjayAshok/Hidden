import click
import pandas as pd
import numpy as np
import os
from models import get_model
from iid_modeling import get_xydf, do_model_fit

results_dir = os.getenv("RESULTS_DIR")
data_dir = os.getenv("DATA_DIR")



@click.command()
@click.option("--task_datasets", type=(str, str), required=True, multiple=True)
@click.option("--model_save_name", type=str, default=None)
@click.option('--random_sample_train_per', type=int, default=None)
@click.option('--random_sample_test_per', type=int, default=None)
@click.option('--random_seed', type=int, default=42)
@click.option('--model_kind', type=click.Choice(['linear', 'mlp', 'transformer'], case_sensitive=False), default="linear")
@click.option('--strict_w_dataset', type=bool, default=False)
@click.option('--strict_w_task', type=bool, default=False)
@click.option('--mix_iid_n', type=int, default=0)
def main(task_datasets, model_save_name, random_sample_train_per, random_sample_test_per, random_seed, model_kind, strict_w_task, strict_w_dataset, mix_iid_n):
    assert len(task_datasets) > 1, "Must provide at least two task-dataset pairs for OOD training"
    np.random.seed(random_seed)
    
    data = {}
    for task, dataset in task_datasets:
        X_train, y_train, train_df = get_xydf(task, dataset, model_save_name, "train", random_sample_train_per, random_seed=random_seed)
        X_test, y_test, test_df = get_xydf(task, dataset, model_save_name, "test", random_sample_test_per, random_seed=random_seed)
        internal_data = {"X_train": X_train, "y_train": y_train, "train_df": train_df, "X_test": X_test, "y_test": y_test, "test_df": test_df}
        data[(task, dataset)] = internal_data        
    
    if strict_w_dataset or strict_w_task:
        all_datasets = set([dataset for task, dataset in task_datasets])
        all_tasks = set([task for task, dataset in task_datasets])
        if strict_w_task:
            assert len(all_tasks) > 1, "Strict w task requires more than one task"
        if strict_w_dataset:
            assert len(all_datasets) > 1, "Strict w dataset requires more than one dataset"
        if strict_w_dataset and strict_w_task:
            for item in task_datasets:
                others = [x for x in task_datasets if x != item]
                flag = False
                for other in others:
                    if other[0] != item[0] and other[1] != item[1]:
                        flag = True
                        break
                assert flag, f"Strict w task and dataset does not have viable training data for {item} on args: {task_datasets}"

    
    for taskdata in data:
        task = taskdata[0]
        dataset = taskdata[1]
        print(f"Fitting model for test: {taskdata}")
        model = get_model(model_kind)
        train_pred, test_pred, train_df, test_df, test_acc = fit_one_set(model, data, taskdata, strict_w_dataset, strict_w_task, mix_iid_n)
        print(f"Final Test Accuracy for [TASK]{task}[TASK] [DATASET]{dataset}[DATASET]: {test_acc}")
        del model
    return
    
def fit_one_set(model, data, test_dataset, strict_w_dataset, strict_w_task, mix_iid_n):
    train_datasets = []
    for taskdata in data:
        if taskdata == test_dataset:
            if mix_iid_n > 0:
                max_mix = min(mix_iid_n, len(data[test_dataset]['y_train']))
                mix_indices = np.random.choice(len(data[test_dataset]['y_train']), max_mix, replace=False)
                mix_X = data[test_dataset]['X_train'][mix_indices]
                mix_y = data[test_dataset]['y_train'][mix_indices]
                mix_df = data[test_dataset]['train_df'].iloc[mix_indices].reset_index(drop=True)
                train_datasets.append({"X_train": mix_X, "y_train": mix_y, "train_df": mix_df})
            else:
                pass
            continue
        if strict_w_dataset:
            if taskdata[1] == test_dataset[1]:
                continue
        if strict_w_task:
            if taskdata[0] == test_dataset[0]:
                continue
        train_datasets.append(data[taskdata])

    X_train = np.concatenate([dataset["X_train"] for dataset in train_datasets], axis=0)
    y_train = np.concatenate([dataset["y_train"] for dataset in train_datasets], axis=0)
    train_df = pd.concat([dataset["train_df"] for dataset in train_datasets], axis=0, ignore_index=True)
    del train_datasets
    X_test = data[test_dataset]["X_test"]
    y_test = data[test_dataset]["y_test"]
    test_df = data[test_dataset]["test_df"]
    train_pred, test_pred, test_acc = do_model_fit(model, X_train, y_train, X_test, y_test)
    print(f"Test Base Rate for [TASK]{test_dataset[0]}[TASK] [DATASET]{test_dataset[1]}[DATASET]: {np.mean(y_test)}")
    return train_pred, test_pred, train_df, test_df, test_acc
    

if __name__ == "__main__":
    main()