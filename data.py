import pandas as pd
from datasets import load_dataset
import numpy as np
import os

np.random.seed(42)

data_dir = os.environ["DATA_DIR"]
for subdir in ["base", "unanswerable", "confidence", "toxicity_avoidance", "jailbreak"]:
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

    train = get_healthver(f"{data_dir}/raw/healthver/healthver_train.csv")
    valid = get_healthver(f"{data_dir}/raw/healthver/healthver_dev.csv")
    test = get_healthver(f"{data_dir}/raw/healthver/healthver_test.csv")
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
    train = get(f"{data_dir}/raw/known_unknown/train.jsonl")
    valid = get(f"{data_dir}/raw/known_unknown/dev.jsonl")
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

def process_real_toxicity_prompts():
    ds = load_dataset("allenai/real-toxicity-prompts")
    df = ds["train"].to_pandas().sample(frac=0.2).reset_index(drop=True)
    df["text"] = df["prompt"].apply(lambda x: x["text"])
    df = df[["text", "challenging"]]
    df = df.sample(frac=1).reset_index(drop=True)
    train = df.loc[:int(len(df) * 0.8)]
    valid = df.loc[int(len(df) * 0.8):].reset_index(drop=True)
    train["idx"] = train.index
    valid["idx"] = valid.index
    train.to_csv(f"{data_dir}/base/real_toxicity_prompts_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/real_toxicity_prompts_test.csv", index=False)
    return train, valid

def process_toxic_chat():
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    def proc_df(df):
        df["text"] = df["user_input"]
        df["idx"] = df.index
        return df[["text", "toxicity", "jailbreaking", "idx"]]
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["test"].to_pandas())
    train.to_csv(f"{data_dir}/base/toxic_chat_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/toxic_chat_test.csv", index=False)

def process_qnota():
    # ambiguous, futuristic, unmeasurable, incorrect
    columns = ["incomplete_questions", "ambiguous_questions", "futuristic_questions", "unmeasurable_questions", "incorrect_questions"]


class ToxicityAvoidance:
    @staticmethod
    def setup_toxic_chat():
        train = pd.read_csv(f"{data_dir}/base/toxic_chat_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/toxic_chat_test.csv")
        train.to_csv(f"{data_dir}/toxicity_avoidance/toxic_chat_train.csv", index=False)
        valid.to_csv(f"{data_dir}/toxicity_avoidance/toxic_chat_test.csv", index=False)
        return train, valid
    
    @staticmethod
    def setup_real_toxicity_prompts():
        train = pd.read_csv(f"{data_dir}/base/real_toxicity_prompts_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/real_toxicity_prompts_test.csv")
        train.to_csv(f"{data_dir}/toxicity_avoidance/real_toxicity_prompts_train.csv", index=False)
        valid.to_csv(f"{data_dir}/toxicity_avoidance/real_toxicity_prompts_test.csv", index=False)
        return train, valid


class Jailbreak:
    @staticmethod
    def setup_toxic_chat():
        train = pd.read_csv(f"{data_dir}/base/toxic_chat_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/toxic_chat_test.csv")
        train["label"] = train["jailbreaking"]
        valid["label"] = valid["jailbreaking"]
        train.to_csv(f"{data_dir}/jailbreak/toxic_chat_train.csv", index=False)
        valid.to_csv(f"{data_dir}/jailbreak/toxic_chat_test.csv", index=False)
        return train, valid

class Unanswerable:
    @staticmethod
    def setupsquad():
        train = pd.read_csv(f"{data_dir}/base/squad_train.csv").sample(12_000).reset_index(drop=True)
        valid = pd.read_csv(f"{data_dir}/base/squad_test.csv")
        def proc_df(df):
            df["text"] = "The following question is either TRUE or FALSE. Which is it?\n" + df["text"] + "\nAnswer: "
            df["label"] = df["unanswerable"].astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        train.to_csv(f"{data_dir}/unanswerable/squad_train.csv", index=False)
        valid.to_csv(f"{data_dir}/unanswerable/squad_test.csv", index=False)
        return train, valid
    
    @staticmethod
    def setupqnota():
        files = ["incomplete_questions", "futuristic_questions", "unmeasurable_questions"]
        columns = ["idx", "group_idx", "type", "text", "label"]
        data = []
        id_it = 0
        group_it = 0
        for file in files:
            df = pd.read_json(f"{data_dir}/raw/qnota/{file}.json")
            if file == "unmeasurable_questions":
                df[file] = df["non_quantifiable_questions"] 
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
        train["label"] = train["label"].astype(int)
        valid["label"] = valid["label"].astype(int)
        # save to data/unanswerable/qnota_train.csv
        train.to_csv(f"{data_dir}/unanswerable/qnota_train.csv", index=False)
        valid.to_csv(f"{data_dir}/unanswerable/qnota_test.csv", index=False)
        return train, valid

    @staticmethod
    def setuphealthver():
        # will set up healthver for the NIE task
        train = pd.read_csv(f"{data_dir}/base/healthver_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/healthver_test.csv")
        def proc_df(df):
            df["text"] = "The following claim is either TRUE or FALSE. Which is it?\n" + df["text"] + "\nAnswer: "
            df["label"] = df["unanswerable"].astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        train.to_csv(f"{data_dir}/unanswerable/healthver_train.csv", index=False)
        valid.to_csv(f"{data_dir}/unanswerable/healthver_test.csv", index=False)

    @staticmethod
    def setupselfaware():
        train = pd.read_csv(f"{data_dir}/base/selfaware_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/selfaware_test.csv")
        def proc_df(df):
            df["text"] = "The following question is either TRUE or FALSE. Which is it?\n" + df["text"] + "\nAnswer: "
            df["label"] = df["unanswerable"].astype(int)
            wouldyous = df['text'].apply(lambda x: "would you rather" in x.lower())
            df = df[~wouldyous]
            df = df.reset_index(drop=True)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        train.to_csv(f"{data_dir}/unanswerable/selfaware_train.csv", index=False)
        valid.to_csv(f"{data_dir}/unanswerable/selfaware_test.csv", index=False)


def mmlu_choices_to_text(choices):
    text = ""
    options = ["A", "B", "C", "D"]
    #options = [0, 1, 2, 3]
    for i, choice in enumerate(choices):
        text += f"\nOption {options[i]}: {choice}"
    return text

def mmlu_answer_to_letter(answer):
    return ["A", "B", "C", "D"][answer]


class Confidence:
    @staticmethod
    def setupmmlu(k=2):
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
                    prompt += f"\nQuestion: {train_subjects.loc[j, 'question']}{mmlu_choices_to_text(train_subjects.loc[j, 'choices'])}\nAnswer: {mmlu_answer_to_letter(train_subjects.loc[j, 'answer'])} [STOP]"
                df.loc[i, "text"] = f"{system_prompt}\n{prompt}\nQuestion: {df.loc[i, 'question']}{mmlu_choices_to_text(df.loc[i, 'choices'])}\nAnswer: "
                df.loc[i, "label"] = mmlu_answer_to_letter(df.loc[i, "answer"])
            return df[["idx", "text", "label"]]
        train_df = proc_df(train)
        valid = proc_df(valid)
        train_df.to_csv(f"{data_dir}/confidence/mmlu_train.csv", index=False)
        valid.to_csv(f"{data_dir}/confidence/mmlu_test.csv", index=False)
        return train_df, valid

def process_all():
    process_squad()
    process_healthver()
    process_selfaware()
    process_known_unkown()
    process_mmlu()
    process_real_toxicity_prompts()
    process_toxic_chat()

def setup_all():
    Unanswerable.setuphealthver()
    Unanswerable.setupselfaware()
    Unanswerable.setupsquad()
    Unanswerable.setupqnota()
    ToxicityAvoidance.setup_toxic_chat()
    ToxicityAvoidance.setup_real_toxicity_prompts()
    Jailbreak.setup_toxic_chat()
    Jailbreak.setup_toxic_chat()
    Confidence.setupmmlu()


if __name__ == "__main__":
    process_all()
    setup_all()