import pandas as pd
from datasets import load_dataset
import numpy as np
import os

np.random.seed(42)

data_dir = "data"
for subdir in ["base", "unanswerable", "confidence"]:
    if not os.path.exists(f"{data_dir}/{subdir}/"):
        os.makedirs(f"{data_dir}/{subdir}/")


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
        df["text"] = "Context: " + df["context"] + "\nQuestion: " + df["question"]
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
        df["text"] = "Evidence: " + df["evidence"] + "\nClaim: " + df["claim"]
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
    def proc_df(df):
        df["idx"] = df.index
        df["choices"]  = df["choices"].apply(lambda x: x.tolist())
        return df
    train = proc_df(train)
    valid = proc_df(valid)
    train.to_csv(f"{data_dir}/base/mmlu_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/mmlu_test.csv", index=False)
    return train, valid

def process_qnota():
    # ambiguous, futuristic, unmeasurable, incorrect
    columns = ["incomplete_questions", "ambiguous_questions", "futuristic_questions", "unmeasurable_questions", "incorrect_questions"]
    pass


def tmp_setupqnota():
    files = ["incomplete_questions", "futuristic_questions", "unmeasurable_questions"]
    columns = ["idx", "group_idx", "type", "text", "unanswerable"]
    data = []
    id_it = 0
    group_it = 0
    for file in files:
        df = pd.read_csv(f"data/raw/qnota/{file}.csv")
        for i, row in df.iterrows():
            unanswerable = row[file]['u']
            answerable = row[file]['a']
            data.append([id_it, group_it, file, answerable, False])
            id_it += 1
            data.append([id_it, group_it, file, unanswerable, True])
            id_it += 1
            group_it += 1
    df = pd.DataFrame(data, columns=columns)
    train = df.sample(frac=0.8)
    valid = df.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    # save to data/unanswerable/qnota_train.csv
    train.to_csv(f"{data_dir}/unanswerable/qnota_train.csv", index=False)
    valid.to_csv(f"{data_dir}/unanswerable/qnota_test.csv", index=False)


def mmlu_choices_to_text(choices):
    text = ""
    options = ["A", "B", "C", "D"]
    #options = [0, 1, 2, 3]
    for i, choice in enumerate(choices):
        text += f"\nOption {options[i]}: {choice}"
    return text

def mmlu_answer_to_letter(answer):
    return ["A", "B", "C", "D"][answer]

def tmp_setupmmlu(k=2):
    train = pd.read_csv(f"{data_dir}/base/mmlu_train.csv")
    valid = pd.read_csv(f"{data_dir}/base/mmlu_test.csv")
    system_prompt = "Answer the following MCQ by providing the correct option"
    def proc_df(df):
        df["choices"] = df["choices"].apply(eval)
        for i in range(len(df)):
            subject = df.loc[i, "subject"]
            train_subjects = train[(train["subject"] == subject) & (train["idx"] != df.loc[i, "idx"])].reset_index(drop=True)
            train_subjects = train_subjects.sample(k).reset_index(drop=True)
            prompt = ""
            for j in range(k):
                prompt += f"\nQuestion: {train_subjects.loc[j, 'question']}{mmlu_choices_to_text(train_subjects.loc[j, 'choices'])}\nAnswer: {train_subjects.loc[j, 'answer']} [STOP]"
            df.loc[i, "text"] = f"{system_prompt}\n{prompt}\nQuestion: {df.loc[i, 'question']}{mmlu_choices_to_text(df.loc[i, 'choices'])}\nAnswer: "
            df.loc[i, "label"] = mmlu_answer_to_letter(df.loc[i, "answer"])
        return df[["idx", "text", "label"]]
    train_df = proc_df(train)
    valid = proc_df(valid)
    train_df.to_csv(f"{data_dir}/confidence/train.csv", index=False)
    valid.to_csv(f"{data_dir}/confidence/test.csv", index=False)
    return 


def tmp_setupsquad():
    train = pd.read_csv(f"{data_dir}/base/squad_train.csv")
    valid = pd.read_csv(f"{data_dir}/base/squad_test.csv")
    def proc_df(df):
        df["text"] = "The following question is either TRUE or FALSE. Which is it?\n" + df["text"] + "\nAnswer: "
        df["label"] = df["unanswerable"].astype(int)
        return df[["idx", "text", "label"]]
    train = proc_df(train)
    valid = proc_df(valid)
    train.to_csv(f"{data_dir}/unanswerable/squad_train.csv", index=False)
    valid.to_csv(f"{data_dir}/unanswerable/squad_test.csv", index=False)


def tmp_setupselfaware():
    train = pd.read_csv(f"{data_dir}/base/selfaware_train.csv")
    valid = pd.read_csv(f"{data_dir}/base/selfaware_test.csv")
    def proc_df(df):
        df["text"] = "The following question is either TRUE or FALSE. Which is it?\n" + df["text"] + "\nAnswer: "
        df["label"] = df["unanswerable"].astype(int)
        return df[["idx", "text", "label"]]
    train = proc_df(train)
    valid = proc_df(valid)
    train.to_csv(f"{data_dir}/unanswerable/selfaware_train.csv", index=False)
    valid.to_csv(f"{data_dir}/unanswerable/selfaware_test.csv", index=False)


def tmp_setuphealthver():
    # will set up healthver for the NIE task
    train = pd.read_csv("data/base/healthver_train.csv")
    valid = pd.read_csv("data/base/healthver_test.csv")
    def proc_df(df):
        df["text"] = "The following claim is either TRUE or FALSE. Which is it?\n" + df["text"] + "\nAnswer: "
        df["label"] = df["unanswerable"].astype(int)
        return df[["idx", "text", "label"]]
    train = proc_df(train)
    valid = proc_df(valid)
    train.to_csv("data/unanswerable/healthver_train.csv", index=False)
    valid.to_csv("data/unanswerable/healthver_test.csv", index=False)

if __name__ == "__main__":
    process_squad()
    process_healthver()
    process_selfaware()
    process_known_unkown()
    process_mmlu()
    tmp_setuphealthver()
    tmp_setupsquad()
    tmp_setupselfaware()
    tmp_setupmmlu()