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
@click.option("--experiment", type=click.Choice(["probe_iid", "probe_ood", "fewshot_corr" ,"fewshot_pred"], case_sensitive=False), default="probe_iid")
def main(experiment):
    base_path = os.path.join(logdir, experiment)
    report_file = report_dir + f"/{experiment}.csv"
    if experiment == "probe_iid":
        df = do_iid_probe(base_path)
    if experiment == "fewshot_pred":
        df = do_fewshot_pred(base_path)
    elif experiment == "probe_ood":
        df = do_ood_probe(base_path)
    elif experiment == "fewshot_corr":
        df = do_fewshot_corr(base_path)
    else:
        raise NotImplementedError
    df.to_csv(report_file, index=False)


def do_iid_probe(base_path):
    model_options = os.listdir(base_path)
    if len(model_options) == 0:
        print(f"No models found in {base_path}. Exiting ...")
        return
    
    data = []
    columns = ["model", "task", "dataset", "model_kind", "n_train", "n_test", "random_seed", "accuracy", "test_base_rate"]

    for model_save_name in model_options:
        task_options = os.listdir(os.path.join(base_path, model_save_name))
        if len(task_options) == 0:
            print(f"No tasks found in {base_path}/{model_save_name}. Skipping ...")
            continue
        for task in task_options:
            dataset_options = os.listdir(os.path.join(base_path, model_save_name, task))
            if len(dataset_options) == 0:
                print(f"No datasets found in {base_path}/{model_save_name}/{task}. Skipping ...")
                continue
            for dataset in dataset_options:
                model_kind_options = os.listdir(os.path.join(base_path, model_save_name, task, dataset))
                if len(model_kind_options) == 0:
                    print(f"No model kinds found in {base_path}/{model_save_name}/{task}/{dataset}. Skipping ...")
                    continue
                for model_kind in model_kind_options:
                    file_options = os.listdir(os.path.join(base_path, model_save_name, task, dataset, model_kind))
                    if len(file_options) == 0:
                        print(f"No files found in {base_path}/{model_save_name}/{task}/{dataset}/{model_kind}. Skipping ...")
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
                            print(f"No accuracies found in {filepath}. Make sure its still running and there hasn't been a more fundamental issue. Skipping ...")
                            continue
                        if len(accuracies) > 1:
                            print(f"Multiple accuracies found in {filepath}. Using the last one ...")
                        accuracy = accuracies[-1]
                        # get the base rate of the test set, it is in a line in the format Base rate: {y_train.mean()} (Train), {y_test.mean()} (Test)
                        base_rate_lines = [line for line in lines if "Base rate: " in line]
                        if len(base_rate_lines) == 0:
                            print(f"No base rate found in {filepath}. Make sure its still running and there hasn't been a more fundamental issue. Skipping ...")
                            continue
                        if len(base_rate_lines) > 1:
                            print(f"Multiple base rates found in {filepath}. Using the last one ...")
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
                                print(f"Could not find train or test inference files for {model_save_name}, {task}, {dataset}. Skipping the autoinference for n_train and n_test ...")
                        data.append([model_save_name, task, dataset, model_kind, n_train, n_test, random_seed, accuracy, base_rate])
    df = pd.DataFrame(data, columns=columns)
    df['baseline'] = df['test_base_rate'].apply(lambda x: max(x, 1-x))
    df['advantage'] = df['accuracy'] - df['baseline']
    return df


def do_ood_probe(base_path):
    model_options = os.listdir(base_path)
    if len(model_options) == 0:
        print(f"No models found in {base_path}. Exiting ...")
        return
    
    data = []
    columns = ["model", "task", "dataset", "run_name_outer", "run_name_inner", "model_kind", "n_train", "random_seed", "accuracy", "test_base_rate"]
    # $LOG_DIR/probe_ood/$model_save_name/$run_name/$model_kind/train_${random_sample_train_per}-seed_${random_seed}.log
    for model_save_name in model_options:
        for run_name in os.listdir(os.path.join(base_path, model_save_name)):
            run_name_outer, run_name_inner = run_name.split("_")
            model_kind_options = os.listdir(os.path.join(base_path, model_save_name, run_name))
            if len(model_kind_options) == 0:
                print(f"No model kinds found in {base_path}/{model_save_name}/{run_name}. Skipping ...")
                continue          
            for model_kind in model_kind_options:
                file_options = os.listdir(os.path.join(base_path, model_save_name, run_name, model_kind))
                if len(file_options) == 0:
                    print(f"No files found in {base_path}/{model_save_name}/{run_name}/{model_kind}. Skipping ...")
                    continue
                for file in file_options:
                    filedata = file.split(".")[0].split("-")
                    filepath = os.path.join(base_path, model_save_name, run_name, model_kind, file)
                    assert "train" in filedata[0] and "seed" in filedata[1] and len(filedata) == 2
                    n_train = int(filedata[0].split("_")[1])
                    random_seed = int(filedata[1].split("_")[1])
                    with open(filepath, "r") as f:
                        lines = f.readlines()
                    accuracy_lines = [line for line in lines if "Final Test Accuracy for" in line]
                    if len(accuracy_lines) == 0:
                        print(f"No accuracies found in {filepath}. Make sure its still running and there hasn't been a more fundamental issue. Skipping ...")
                        continue
                    tasks = []
                    datasets = []
                    accuracies = []
                    for line in accuracy_lines:
                        task = line.split("[TASK]")[1].split("[TASK]")[0]
                        dataset = line.split("[DATASET]")[1].split("[DATASET]")[0]
                        accuracy = float(line.split(": ")[1])
                        tasks.append(task)
                        datasets.append(dataset)
                        accuracies.append(accuracy)
                    base_rate_lines = [line for line in lines if "Base Rate for" in line]                    
                    if len(base_rate_lines) == 0:
                        print(f"No base rate found in {filepath}. Make sure its still running and there hasn't been a more fundamental issue. Skipping ...")
                        continue
                    if len(base_rate_lines) != len(accuracy_lines):
                        print(f"Base rate and accuracy lines do not match in {filepath}. {len(base_rate_lines), len(accuracy_lines)} Skipping ...")
                        continue
                    base_rates = [float(line.split(": ")[1]) for line in base_rate_lines]
                    for task, dataset, accuracy, base_rate in zip(tasks, datasets, accuracies, base_rates):
                        data.append([model_save_name, task, dataset, run_name_outer, run_name_inner, model_kind, n_train, random_seed, accuracy, base_rate])
                        
    df = pd.DataFrame(data, columns=columns)
    df['baseline'] = df['test_base_rate'].apply(lambda x: max(x, 1-x))
    df['advantage'] = df['accuracy'] - df['baseline']
    return df
    

def do_fewshot_pred(base_path):
    # this one won't use base_path
    base_path = data_dir + "/fewshot_eval"
    model_options = os.listdir(base_path)
    if len(model_options) == 0:
        print(f"No models found in {base_path}. Exiting ...")
        return
    data = []
    columns = ["model", "task", "dataset", "idx", "label", "fewshot_pred"]
    for model_save_name in model_options:
        task_options = os.listdir(os.path.join(base_path, model_save_name))
        if len(task_options) == 0:
            print(f"No tasks found in {base_path}/{model_save_name}. Skipping ...")
            continue
        for task in task_options:
            dataset_options = os.listdir(os.path.join(base_path, model_save_name, task))
            if len(dataset_options) == 0:
                print(f"No datasets found in {base_path}/{model_save_name}/{task}. Skipping ...")
                continue
            for dataset_file in dataset_options:
                dataset_name = dataset_file.split("_")[0]
                if "indosentiment" in dataset_name:
                    df = pd.read_csv(f"{base_path}/{model_save_name}/{task}/{dataset_file}", lineterminator="\n")
                else:
                    df = pd.read_csv(f"{base_path}/{model_save_name}/{task}/{dataset_file}")
                flag = False
                for needed_column in ["idx", "label", "fewshot_pred"]:
                    if needed_column not in df.columns:
                        print(f"Missing column {needed_column} in {base_path}/{model_save_name}/{task}/{dataset_file} with columns {df.columns} Skipping ...")
                        flag = True
                        break
                if flag:
                    continue
                label_col = "label"
                for i, row in df.iterrows():

                    data.append([model_save_name, task, dataset_name, row["idx"], row[label_col], row["fewshot_pred"]])
    df = pd.DataFrame(data, columns=columns)
    df["fewshot_correct"] = df["label"] == df["fewshot_pred"]
    return df


def do_fewshot_corr(base_path):
    # this one won't use base path
    fewshot_file = report_dir+"/fewshot_pred.csv"
    if not os.path.exists(fewshot_file):
        print(f"Fewshot file not found in {fewshot_file}. Exiting ...")
        return
    fewshot_df = pd.read_csv(fewshot_file)
    base_path = results_dir
    data = []
    columns = ["model", "task", "dataset", "run_name", "model_kind", "train_config", "correlation"]
    model_options = os.listdir(base_path)
    if len(model_options) == 0:
        print(f"No models found in {base_path}. Exiting ...")
        return
    for model_save_name in model_options:
        prediction_dir = os.path.join(base_path, model_save_name, "predictions")
        if not os.path.exists(prediction_dir):
            print(f"No predictions found in {prediction_dir}. Skipping ...")
            continue
        run_name_base_options = os.listdir(prediction_dir)
        if len(run_name_base_options) == 0:
            print(f"No run names found in {prediction_dir}. Skipping ...")
            continue
        for run_name_base in run_name_base_options:
            tasks = os.listdir(os.path.join(prediction_dir, run_name_base))
            if len(tasks) == 0:
                print(f"No tasks found in {prediction_dir}/{run_name_base}. Skipping ...")
                continue
            for task in tasks:
                datasets = os.listdir(os.path.join(prediction_dir, run_name_base, task))
                if len(datasets) == 0:
                    print(f"No datasets found in {prediction_dir}/{run_name_base}/{task}. Skipping ...")
                    continue
                condition = (fewshot_df["model"] == model_save_name) & ((fewshot_df["task"] == task) & (fewshot_df["dataset"] == dataset))
                fewshot_subset = fewshot_df[condition]
                if len(fewshot_subset) == 0:
                    print(f"No fewshot data found for {model_save_name}, {task}, {dataset}. Skipping ...")
                    continue
                for dataset in datasets:
                    model_kinds = os.listdir(os.path.join(prediction_dir, run_name_base, task, dataset))
                    if len(model_kinds) == 0:
                        print(f"No model kinds found in {prediction_dir}/{run_name_base}/{task}/{dataset}. Skipping ...")
                        continue
                    for model_kind in model_kinds:
                        train_configs = os.listdir(os.path.join(prediction_dir, run_name_base, task, dataset, model_kind))
                        if len(train_configs) == 0:
                            print(f"No train configs found in {prediction_dir}/{run_name_base}/{task}/{dataset}/{model_kind}. Skipping ...")
                            continue
                        for train_config in train_configs:
                            test_csv = os.path.join(prediction_dir, run_name_base, task, dataset, model_kind, train_config, "test_pred.csv")
                            if not os.path.exists(test_csv):
                                print(f"No test csv found in {test_csv}. Skipping ...")
                                continue
                            test_df = pd.read_csv(test_csv)
                            flag = False
                            if len(fewshot_subset) != len(test_df):
                                flag = True
                                print(f"Fewshot and test data lengths do not match for {model_save_name}, {task}, {dataset}, {model_kind}, {train_config}. Trying manual resolution...")
                            if len(fewshot_subset) == len(test_df):
                                if not (fewshot_subset.label == test_df.label).all():
                                    flag = True
                                    print(f"Fewshot and test data labels do not match for {model_save_name}, {task}, {dataset}, {model_kind}, {train_config}. Trying manual resolution ...")
                            if flag:
                                test_df["fewshot_correct"] = None
                                n_zero = 0
                                n_more_than_one = 0
                                for i, row in test_df.iterrows():
                                    idx = row["idx"]
                                    fewshot_idx_row = fewshot_subset[fewshot_subset["idx"] == idx].reset_index(drop=True)
                                    if len(fewshot_idx_row) == 0:
                                        n_zero += 1
                                        continue
                                    if len(fewshot_idx_row) > 1:
                                        n_more_than_one += 1
                                        continue
                                    fewshot_row = fewshot_idx_row.loc[0]
                                    test_df.loc[i, "fewshot_correct"] = fewshot_row["fewshot_correct"]
                                if test_df["fewshot_correct"].isnull().sum() > 0:
                                    print(f"There are {test_df['fewshot_correct'].isnull().sum()} null values in {model_save_name}, {task}, {dataset}, {model_kind}, {train_config}. Be warned ...")
                                if n_zero > 0:
                                    print(f"\tFound {n_zero} zero matches in {model_save_name}, {task}, {dataset}, {model_kind}, {train_config}. Be warned ...")
                                if n_more_than_one > 0:
                                    print(f"\tFound {n_more_than_one} more than one matches in {model_save_name}, {task}, {dataset}, {model_kind}, {train_config}. Be warned ...")
                                test_df["probe_correct"] = (test_df["probe_prediction"] > 0.5) == test_df["label"].astype(bool)
                                correlation = test_df[["probe_correct", "fewshot_correct"]].corr()["probe_correct"]["fewshot_correct"]
                                data.append([model_save_name, task, dataset, run_name_base, model_kind, train_config, correlation])
    df = pd.DataFrame(data, columns=columns)
    return df



if __name__ == "__main__":
    main()