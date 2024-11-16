import click
import pandas as pd
import os
import warnings

logdir = os.environ["LOG_DIR"]
results_dir = os.environ["RESULTS_DIR"]
data_dir = os.environ["DATA_DIR"]

report_dir = os.path.join(logdir, "reports")
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

@click.command() # get experiment, model_save_name, task, dataset, model_kind from user
@click.option("--experiment", type=click.Choice(["probe_iid", "probe_ood", "inter_iid", "inter_ood", "fewshot_pred"], case_sensitive=False), default="probe_iid")
def main(experiment):
    base_path = os.path.join(logdir, experiment)
    report_file = report_dir + f"/{experiment}.csv"
    if experiment == "probe_iid":
        df = do_iid_probe(base_path, report_file)
    elif experiment == "probe_ood":
        raise NotImplementedError
    else:
        raise NotImplementedError
    df.to_csv(report_file, index=False)


def do_iid_probe(base_path):
    model_options = os.listdir(base_path)
    if len(model_options) == 0:
        warnings.warn(f"No models found in {base_path}. Exiting ...")
        return
    
    data = []
    columns = ["model", "task", "dataset", "model_kind", "n_train", "n_test", "random_seed", "accuracy", "test_base_rate"]

    for model_save_name in model_options:
        task_options = os.listdir(os.path.join(base_path, model_save_name))
        if len(task_options) == 0:
            warnings.warn(f"No tasks found in {base_path}/{model_save_name}. Skipping ...")
            continue
        for task in task_options:
            dataset_options = os.listdir(os.path.join(base_path, model_save_name, task))
            if len(dataset_options) == 0:
                warnings.warn(f"No datasets found in {base_path}/{model_save_name}/{task}. Skipping ...")
                continue
            for dataset in dataset_options:
                model_kind_options = os.listdir(os.path.join(base_path, model_save_name, task, dataset))
                if len(model_kind_options) == 0:
                    warnings.warn(f"No model kinds found in {base_path}/{model_save_name}/{task}/{dataset}. Skipping ...")
                    continue
                for model_kind in model_kind_options:
                    file_options = os.listdir(os.path.join(base_path, model_save_name, task, dataset, model_kind))
                    if len(file_options) == 0:
                        warnings.warn(f"No files found in {base_path}/{model_save_name}/{task}/{dataset}/{model_kind}. Skipping ...")
                        continue
                    for file in file_options:
                        filedata = file.split(".")[0].split("-")
                        filepath = os.path.join(base_path, model_save_name, task, dataset, model_kind, file)
                        assert "train" in filedata[0] and "test" in filedata[1] and "seed" in filedata[2] and len(filedata) == 3
                        # all iid format train_${random_sample_train}-test_${random_sample_test}-seed_${random_seed}.log
                        n_train = int(filedata[0].split("_")[1])
                        n_test = int(filedata[1].split("_")[1])
                        random_seed = int(filedata[2].split("_")[1])
                        # read the file and get all lines. search for Final Test Accuracy: pattern and get all instances of it
                        with open(filepath, "r") as f:
                            lines = f.readlines()
                        accuracies = [float(line.split(": ")[1]) for line in lines if "Final Test Accuracy: " in line]
                        if len(accuracies) == 0:
                            warnings.warn(f"No accuracies found in {filepath}. Make sure its still running and there hasn't been a more fundamental issue. Skipping ...")
                            continue
                        if len(accuracies) > 1:
                            warnings.warn(f"Multiple accuracies found in {filepath}. Using the last one ...")
                        accuracy = accuracies[-1]
                        # get the base rate of the test set, it is in a line in the format Base rate: {y_train.mean()} (Train), {y_test.mean()} (Test)
                        base_rate_lines = [line for line in lines if "Base rate: " in line]
                        if len(base_rate_lines) == 0:
                            warnings.warn(f"No base rate found in {filepath}. Make sure its still running and there hasn't been a more fundamental issue. Skipping ...")
                            continue
                        if len(base_rate_lines) > 1:
                            warnings.warn(f"Multiple base rates found in {filepath}. Using the last one ...")
                        base_rate_str = base_rate_lines[-1].split(",")[1].split("(")[0].strip()
                        base_rate = float(base_rate_str)
                        # now try to open the train and test files and check whether it has n_train and n_test points
                        try:
                            train_df = pd.read_csv(f"{results_dir}/{model_save_name}/{task}/{dataset}_train_inference.csv")
                            if len(train_df) < n_train:
                                n_train = len(train_df)
                            test_df = pd.read_csv(f"{results_dir}/{model_save_name}/{task}/{dataset}_test_inference.csv")
                            if len(test_df) < n_test:
                                n_test = len(test_df)
                        except:
                            if "indosentiment" in dataset:
                                train_df = pd.read_csv(f"{results_dir}/{model_save_name}/{task}/{dataset}_train_inference.csv", lineterminator="\n")
                                if len(train_df) < n_train:
                                    n_train = len(train_df)
                                test_df = pd.read_csv(f"{results_dir}/{model_save_name}/{task}/{dataset}_test_inference.csv", lineterminator="\n")
                                if len(test_df) < n_test:
                                    n_test = len(test_df)
                            else:
                                warnings.warn(f"Could not find train or test inference files for {model_save_name}, {task}, {dataset}. Skipping the autoinference for n_train and n_test ...")
                        data.append([model_save_name, task, dataset, model_kind, n_train, n_test, random_seed, accuracy, base_rate])
    df = pd.DataFrame(data, columns=columns)
    df['baseline'] = df['test_base_rate'].apply(lambda x: max(x, 1-x))
    df['advantage'] = df['accuracy'] - df['baseline']
    return df


def do_fewshot_pred(base_path):
    # this one won't use base_path
    base_path = data_dir + "/fewshot_pred"
    model_options = os.listdir(base_path)
    if len(model_options) == 0:
        warnings.warn(f"No models found in {base_path}. Exiting ...")
        return
    data = []
    columns = ["model", "task", "dataset", "text", "label", "fewshot_pred"]
    for model_save_name in model_options:
        task_options = os.listdir(os.path.join(base_path, model_save_name))
        if len(task_options) == 0:
            warnings.warn(f"No tasks found in {base_path}/{model_save_name}. Skipping ...")
            continue
        for task in task_options:
            dataset_options = os.listdir(os.path.join(base_path, model_save_name, task))
            if len(dataset_options) == 0:
                warnings.warn(f"No datasets found in {base_path}/{model_save_name}/{task}. Skipping ...")
                continue
            for dataset_file in dataset_options:
                df = pd.read_csv(f"{base_path}/{model_save_name}/{task}/{dataset_file}.csv")
                dataset_name = dataset_file.split("_")[0]
                for needed_column in ["text", "label", "fewshot_pred"]:
                    if needed_column not in df.columns:
                        warnings.warn(f"Missing column {needed_column} in {base_path}/{model_save_name}/{task}/{dataset_file} with columns {df.columns} Skipping ...")
                        continue
                for i, row in df.iterrows():
                    data.append([model_save_name, task, dataset_name, row["label"], row["fewshot_pred"]])
    df = pd.DataFrame(data, columns=columns)
    return df

if __name__ == "__main__":
    main()