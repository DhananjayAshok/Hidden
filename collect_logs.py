import click
import pandas as pd
import os
import warnings

logdir = os.environ["LOG_DIR"]
report_dir = os.path.join(logdir, "reports")
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

@click.command() # get experiment, model_save_name, task, dataset, model_kind from user
# experiment has options ["probe_iid", "probe_ood", "inter_iid", "inter_ood"]
@click.option("--experiment", type=click.Choice(["probe_iid", "probe_ood", "inter_iid", "inter_ood"], case_sensitive=False), default="probe_iid")
def main(experiment):
    base_path = os.path.join(logdir, experiment)
    report_file = report_dir + f"/{experiment}.csv"
    if experiment == "probe_iid":
        do_iid_probe(base_path, report_file)
    elif experiment == "probe_ood":
        raise NotImplementedError
    else:
        raise NotImplementedError


def do_iid_probe(base_path, report_file):
    model_options = os.listdir(base_path)
    if len(model_options) == 0:
        warnings.warn(f"No models found in {base_path}. Exiting ...")
        return
    
    data = []
    columns = ["model", "task", "dataset", "model_kind", "n_train", "n_test", "random_seed", "accuracy"]

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
                        assert "train" in filedata[0] and "test" in filedata[1] and "seed" in filedata[2] and len(filedata) == 3
                        # all iid format train_${random_sample_train}-test_${random_sample_test}-seed_${random_seed}.log
                        n_train = int(filedata[0].split("_")[1])
                        n_test = int(filedata[1].split("_")[1])
                        random_seed = int(filedata[2].split("_")[1])
                        # read the file and get all lines. search for Final Test Accuracy: pattern and get all instances of it
                        with open(os.path.join(base_path, model_save_name, task, dataset, model_kind, file), "r") as f:
                            lines = f.readlines()
                        accuracies = [float(line.split(": ")[1]) for line in lines if "Final Test Accuracy: " in line]
                        if len(accuracies) == 0:
                            warnings.warn(f"No accuracies found in {file}. Make sure its still running and there hasn't been a more fundamental issue. Skipping ...")
                            continue
                        if len(accuracies) > 1:
                            warnings.warn(f"Multiple accuracies found in {file}. Using the last one ...")
                        accuracy = accuracies[-1]
                        data.append([model_save_name, task, dataset, model_kind, n_train, n_test, random_seed, accuracy])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(report_file, index=False)
    return




if __name__ == "__main__":
    main()