import pandas as pd
from datasets import load_dataset
import numpy as np
import os

np.random.seed(42)

data_dir = "data"
if not os.path.exists(f"{data_dir}/base/"):
    os.makedirs(f"{data_dir}/base/")



def process_squad():
    ds = load_dataset("rajpurkar/squad_v2")
    train = ds["train"].to_pandas()
    valid = ds["validation"].to_pandas()
    def answer_na(x):
        if len(x) == 0:
            return None
        else:
            return x[0]
    def proc_df(df):
        df["text"] = df["context"] + "\nQuestion: " + df["question"]
        df["answer"] = df["answers"].apply(lambda x: answer_na(x["text"]))
        df["unanswerable"] = df["answer"].isna()
        return df[["context", "question", "text", "answer", "unanswerable"]]
    train = proc_df(train)
    valid = proc_df(valid)
    train["idx"] = train.index
    valid["idx"] = valid.index
    train.to_csv(f"{data_dir}/base/squad_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/squad_test.csv", index=False)
    return train, valid

def process_healthver():
    def get_healthver(path):
        df = pd.read_csv(path)
        df["text"] = df["evidence"] + "\nClaim: " + df["claim"]
        df["unanswerable"] = df["label"] == "Neutral"
        return df[["evidence", "claim", "text", "label", "unanswerable"]]

    train = get_healthver("data/raw/healthver/healthver_train.csv")
    valid = get_healthver("data/raw/healthver/healthver_dev.csv")
    test = get_healthver("data/raw/healthver/healthver_test.csv")
    train = pd.concat([train, valid], ignore_index=True)
    train["idx"] = train.index
    test["idx"] = test.index
    train.to_csv(f"{data_dir}/base/healthver_train.csv", index=False)
    test.to_csv(f"{data_dir}/base/healthver_test.csv", index=False)
    return train, test

def process_selfaware():
    ds = load_dataset("JesusCrist/selfAware")
    train = ds["train"].to_pandas()
    train["unanswerable"] = train["answerable"] == False
    df = train[["question", "answer", "unanswerable"]]
    train = df.sample(frac=0.8)
    valid = df.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    train["idx"] = train.index
    valid["idx"] = valid.index
    train["text"] = train["question"]
    valid["text"] = valid["question"]
    train.to_csv(f"{data_dir}/base/selfaware_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/selfaware_test.csv", index=False)
    return train, valid


def process_known_unkown():
    def get(path):
        df = pd.read_json(path, lines=True)
        df["unanswerable"] = df["category"].isin(["unsolved problem", "future unknown", "ambiguous"])
        df["text"] = df["question"]        
        df["id"] = df.index
        df = df[["id", "question", "answer", "unanswerable", "text"]]
        return df
    train = get("data/raw/known_unknown/train.jsonl")
    valid = get("data/raw/known_unknown/dev.jsonl")
    train.to_csv(f"{data_dir}/base/known_unknown_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/known_unknown_test.csv", index=False)
    return train, valid


def process_mmlu():
    ds = load_dataset("cais/mmlu", "all", split=["test", "validation"])
    train = ds[1].to_pandas()
    valid = ds[0].to_pandas()
    train["idx"] = train.index
    valid["idx"] = valid.index
    train.to_csv(f"{data_dir}/base/mmlu_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/mmlu_test.csv", index=False)
    return train, valid