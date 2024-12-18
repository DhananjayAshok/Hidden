import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

sns.set_theme(style="whitegrid")

logdir = os.environ["LOG_DIR"]
reports_path = logdir + "/reports/"
figure_path="figures/"
os.makedirs(figure_path, exist_ok=True)

show_figs = False

def augment_ood(ood_df, iid_df):
    for plot_col in ["accuracy", "advantage"]:
        for i, row in ood_df.iterrows():
            task = row["task"]
            dataset = row["dataset"]
            model = row["model"]
            condition = ((iid_df["task"] == task) & (iid_df["dataset"] == dataset) & (iid_df["model"] == model))
            iid_info_df = iid_df[condition]
            if len(iid_info_df) == 0:
                print(f"Task {task} and {dataset} not found in IID data for model {model}. Skipping...")
                continue
            ood_df.loc[i, f"{plot_col}_iid"] = iid_info_df.iloc[0][plot_col]


def augment_w_fewshot(df, fewshot_agg):
    df["fewshot_accuracy"] = None
    df["fewshot_advantage"] = None
    for model, task, dataset in fewshot_agg.index:
        fewshot_acc = fewshot_agg.loc[model, task, dataset]
        condition = ((df["model"] == model) & (df["task"] == task) & (df["dataset"] == dataset))
        df.loc[condition, "fewshot_accuracy"] = fewshot_acc
    df["fewshot_advantage"] = df["fewshot_accuracy"] - df["test_base_rate"]





def save_fig(figpath):
    if show_figs:
        print(f"Save this as: {figpath}")
        plt.show()
    else:
        if not os.path.exists(os.path.dirname(figpath)):
            os.makedirs(os.path.dirname(figpath))
        plt.savefig(figpath)
        plt.clf()


def plot_metric(df, plot_col="advantage", save_suffix="iid", remove_variants=True):
    models = df["model"].unique()
    for model in models:
        model_df = df[df["model"] == model]
        rel_df = model_df
        if remove_variants:
            rel_df = model_df[~model_df["task"].str.contains("_")] 
        sns.boxplot(x="task", y=plot_col, data=rel_df)
        plt.title(f"Hidden Probe Performance on {model}")
        plt.xlabel("Task")
        plt.ylabel(plot_col[0].upper() + plot_col[1:])
        #plt.show()
        save_fig(figure_path + f"{model}/{save_suffix}_{plot_col}.png")

        sns.scatterplot(x=f"fewshot_{plot_col}", y=plot_col, data=rel_df, hue="task")
        plt.title(f"Fewshot vs Hidden Probe Performance for {model}")
        plt.xlabel(f"Fewshot {plot_col[0].upper() + plot_col[1:]}")
        plt.ylabel(plot_col[0].upper() + plot_col[1:])
        #plt.show()
        save_fig(figure_path + f"{model}/{save_suffix}_fewshot_{plot_col}.png")

        for task in model_df["task"].unique():
            if remove_variants and "_" in task:
                continue
            task_df = model_df[model_df["task"] == task]
            task_df = task_df.sort_values("advantage")
            sns.barplot(x="dataset", y=plot_col, data=task_df)
            plt.title(f"Hidden Probe Performance on {task} tasks for {model}")
            plt.xlabel("Dataset")
            plt.ylabel(plot_col[0].upper() + plot_col[1:])
            #plt.show()
            save_fig(figure_path + f"{model}/{save_suffix}_{task}_{plot_col}.png")
        

def plot_iid_ood(iid_df, ood_df, remove_variants=True):
    plot_col = "accuracy"
    for i, row in iid_df.iterrows():
        task = row["task"]
        dataset = row["dataset"]
        model = row["model"]
        condition = ((ood_df["task"] == task) & (ood_df["dataset"] == dataset) & (ood_df["model"] == model))
        ood_info_df = ood_df[condition]
        if len(ood_info_df) == 0:
            print(f"Task {task} and {dataset} not found in OOD data for model {model}. Skipping...")
            continue
        iid_df.loc[i, f"{plot_col}_ood_penalty"] = row[plot_col] - ood_info_df.iloc[0][plot_col]

    
    for model in iid_df["model"].unique():
        model_df = iid_df[iid_df["model"] == model]
        sns.boxplot(x="task", y=f"{plot_col}_ood_penalty", data=model_df)
        plt.title(f"Distribution Shift Penalty on {model} (IID-OOD)")
        plt.xlabel("Task")
        plt.ylabel(f"{plot_col[0].upper() + plot_col[1:]} (IID-OOD)")
        save_fig(figure_path + f"{model}/iid_ood_{plot_col}.png")
        for task in model_df["task"].unique():
            if "_" in task and remove_variants:
                continue
            task_df = model_df[model_df["task"] == task]
            task_df = task_df.sort_values(f"{plot_col}_ood_penalty")
            sns.barplot(x="dataset", y=f"{plot_col}_ood_penalty", data=task_df)
            plt.title(f"Distribution Shift Penalty on {task} tasks for {model} (IID-OOD)")
            plt.xlabel("Dataset")
            plt.ylabel(f"{plot_col[0].upper() + plot_col[1:]} (IID-OOD)")
            save_fig(figure_path + f"{model}/iid_ood_{task}_{plot_col}.png")

def plot_variant(iid_df, plot_col="advantage"):
    for model in iid_df["model"].unique():
        model_df = iid_df[iid_df["model"] == model]
        for task in model_df["task"].unique():
            task_df = model_df[model_df["task"].apply(lambda x: x.startswith(task))]
            if len(task_df) == 0:
                continue
            sns.boxplot(x="task", y=plot_col, data=task_df)
            plt.title(f"Hidden Probe Performance on {task} variants for {model}")
            plt.xlabel("Variant")
            plt.ylabel(plot_col[0].upper() + plot_col[1:])
            save_fig(figure_path + f"{model}/variant_{task}_{plot_col}.png")

if __name__ == "__main__":
    iid_df = pd.read_csv(reports_path + "/probe_iid.csv")
    ood_df = pd.read_csv(reports_path + "/probe_ood.csv")
    fewshot_df = pd.read_csv(reports_path + "/fewshot_pred.csv")
    fewshot_agg = fewshot_df.groupby(["model", "task", "dataset"])["fewshot_correct"].mean()
    augment_ood(ood_df, iid_df)
    augment_w_fewshot(iid_df, fewshot_agg)
    augment_w_fewshot(ood_df, fewshot_agg)
    plot_iid_ood(iid_df, ood_df)
    for plot_col in ["accuracy", "advantage"]:
        plot_metric(iid_df, plot_col)
        plot_metric(ood_df, plot_col, "ood")
        plot_variant(iid_df, plot_col)

