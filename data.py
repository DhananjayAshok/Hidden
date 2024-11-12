import pandas as pd
from datasets import load_dataset
import numpy as np
import os
import re

def remove_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Substitute URLs with an empty string
    cleaned_text = url_pattern.sub('', text)
    return cleaned_text

data_dir = os.environ["DATA_DIR"]
for subdir in ["base", "unanswerable", "confidence", "toxicity_avoidance", "jailbreak"]:
    if not os.path.exists(f"{data_dir}/{subdir}/"):
        os.makedirs(f"{data_dir}/{subdir}/")


def process_nytimes(random_seed=42, save=True):
    df = pd.read_json(f"{data_dir}/raw/nytimes/nytimes_dataset.json")
    df["text"] = df["headline"] + ". " + df["abstract"]
    train = df.sample(frac=0.8, random_state=random_seed)
    valid = df.drop(train.index).reset_index(drop=True)
    train["idx"] = train.index
    valid["idx"] = valid.index
    if save:
        train.to_csv(f"{data_dir}/base/nytimes_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/nytimes_test.csv", index=False)
    return train, valid


def process_agnews():
    ds = load_dataset("SetFit/ag_news")
    def proc_df(df):
        df["idx"] = df.index
        return df
    train = proc_df(ds["train"].to_pandas())
    test = proc_df(ds["test"].to_pandas())
    train.to_csv(f"{data_dir}/base/agnews_train.csv", index=False)
    test.to_csv(f"{data_dir}/base/agnews_test.csv", index=False)

def process_bbcnews():
    ds = load_dataset("SetFit/bbc-news")
    def proc_df(df):
        df["idx"] = df.index
        return df
    train = proc_df(ds["train"].to_pandas())
    test = proc_df(ds["test"].to_pandas())
    train.to_csv(f"{data_dir}/base/bbcnews_train.csv", index=False)
    test.to_csv(f"{data_dir}/base/bbcnews_test.csv", index=False)


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

def process_selfaware(save=True, random_seed=42):
    ds = load_dataset("JesusCrist/selfAware")
    train = ds["train"].to_pandas()
    train["unanswerable"] = train["answerable"] == False
    df = train[["question", "answer", "unanswerable"]]
    train = df.sample(frac=0.8, random_state=random_seed)
    valid = df.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    train["idx"] = train.index
    valid["idx"] = valid.index
    train["text"] = train["question"]
    valid["text"] = valid["question"]
    if save:
        train.to_csv(f"{data_dir}/base/selfaware_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/selfaware_test.csv", index=False)
    return train, valid

def process_known_unkown():
    def get(path):
        df = pd.read_json(path, lines=True)
        df["unanswerable"] = df["category"].isin(["unsolved problem", "future unknown", "ambiguous"])
        df["text"] = df["question"]        
        df["idx"] = df.index
        df = df[["idx", "question", "answer", "unanswerable", "text"]]
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

def process_real_toxicity_prompts(save=True, random_seed=42):
    ds = load_dataset("allenai/real-toxicity-prompts")
    df = ds["train"].to_pandas().sample(frac=0.2, random_state=random_seed).reset_index(drop=True)
    df["text"] = df["prompt"].apply(lambda x: x["text"])
    df = df[["text", "challenging"]]
    df["prompt_only"] = True
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    train = df.loc[:int(len(df) * 0.8)]
    valid = df.loc[int(len(df) * 0.8):].reset_index(drop=True)
    train["idx"] = train.index
    valid["idx"] = valid.index
    if save:
        train.to_csv(f"{data_dir}/base/real_toxicity_prompts_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/real_toxicity_prompts_test.csv", index=False)
    return train, valid

def process_toxic_chat():
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    def proc_df(df):
        df["text"] = df["user_input"]
        df["idx"] = df.index
        df["prompt_only"] = True
        return df[["text", "toxicity", "jailbreaking", "idx", "prompt_only"]]
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["test"].to_pandas())
    train.to_csv(f"{data_dir}/base/toxic_chat_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/toxic_chat_test.csv", index=False)

def process_qnota():
    # ambiguous, futuristic, unmeasurable, incorrect
    columns = ["incomplete_questions", "ambiguous_questions", "futuristic_questions", "unmeasurable_questions", "incorrect_questions"]


def process_amazonreviews(random_seed=32, save=True):
    ds = load_dataset("mteb/amazon_reviews_multi", "en")
    def process_df(df):
        df = df[df["label"].isin([0, 4])]
        df["label"] = df["label"] == 4
        return df[["text", "label"]]
    train = process_df(ds["train"].to_pandas())
    valid = process_df(ds["validation"].to_pandas())
    test = process_df(ds["test"].to_pandas())
    train_df = pd.concat([train, valid], ignore_index=True)
    train_df = train_df.sample(frac=0.25, random_state=random_seed).reset_index(drop=True)
    train_df["idx"] = train_df.index
    test["idx"] = test.index
    if save:
        train_df.to_csv(f"{data_dir}/base/amazonreviews_train.csv", index=False)
        test.to_csv(f"{data_dir}/base/amazonreviews_test.csv", index=False)
    return train_df, test    

def process_yelp(random_seed=32, save=True):
    ds = load_dataset("noahnsimbe/yelp-dataset")
    def proc_df(df):
        df = df[df["label"].isin([0, 2])]
        df["label"] = df["label"] == 2
        return df[["text", "label"]]
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["validation"].to_pandas())
    test = proc_df(ds["test"].to_pandas())
    train_df = pd.concat([train, valid], ignore_index=True)
    train_df = train_df.sample(frac=0.15, random_state=random_seed).reset_index(drop=True)
    train_df["idx"] = train_df.index
    test["idx"] = test.index
    if save:
        train_df.to_csv(f"{data_dir}/base/yelp_train.csv", index=False)
        test.to_csv(f"{data_dir}/base/yelp_test.csv", index=False)
    return train_df, test

def process_twitterfinance():
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    def proc_df(df):
        df["idx"] = df.index
        df["text"] = df["text"].apply(remove_urls)
        df["label"] = df["label"] == 2
        return df[["idx", "text", "label"]]
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["validation"].to_pandas())
    train.to_csv(f"{data_dir}/base/twitterfinance_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/twitterfinance_test.csv", index=False)
    return train, valid

def process_twittermteb():
    ds = load_dataset("mteb/tweet_sentiment_extraction")
    def proc_df(df):
        df["idx"] = df.index
        df["text"] = df["text"].apply(remove_urls)
        df["label"] = df["label_text"] == "positive"
        df = df[df["text"].apply(lambda x: len(x.split())) > 5]
        return df[["idx", "text", "label"]]
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["test"].to_pandas())
    train.to_csv(f"{data_dir}/base/twittermteb_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/twittermteb_test.csv", index=False)
    return train, valid
    
def process_auditor():
    ds = load_dataset("FinanceInc/auditor_sentiment")
    def proc_df(df):
        df["idx"] = df.index
        df["label"] = df["label"] == 2
        return df[["idx", "text", "label"]]
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["test"].to_pandas())
    train.to_csv(f"{data_dir}/base/auditorsentiment_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/auditorsentiment_test.csv", index=False)
    return train, valid

def process_fiqa():
    ds = load_dataset("ChanceFocus/fiqa-sentiment-classification")
    def proc_df(df):
        df["text"] = df["sentence"]
        # get the 75th percentile of the sentiment 'score'
        df["label"] = df['score'] > df['score'].quantile(0.75)
        return df[["idx", "text", "label"]]
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["valid"].to_pandas())
    test = proc_df(ds["test"].to_pandas())
    train_df = pd.concat([train, valid], ignore_index=True).reset_index(drop=True)
    train_df["idx"] = train_df.index
    test["idx"] = test.index
    train_df.to_csv(f"{data_dir}/base/fiqa_train.csv", index=False)
    test.to_csv(f"{data_dir}/base/fiqa_test.csv", index=False)
    return train_df, test

def process_indosentiment(random_seed=42, save=True):
    ds = load_dataset("dipawidia/ecommerce-product-reviews-sentiment")
    df = ds["train"].to_pandas()
    df["text"] = df["translate"]
    df["alt_text"] = df["review"]
    df["label"] = df["sentimen"]
    df = df[["text", "alt_text", "label"]]
    train = df.sample(frac=0.8, random_state=random_seed)
    valid = df.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    train["idx"] = train.index
    valid["idx"] = valid.index
    if save:
        train.to_csv(f"{data_dir}/base/indosentiment_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/indosentiment_test.csv", index=False)
    return train, valid

def process_newsmtc():
    ds = load_dataset("fhamborg/news_sentiment_newsmtsc", trust_remote_code=True)
    def proc_df(df):
        df["text"] = df["sentence"]
        df["label"] = df["polarity"] == 1
        return df[["idx", "text", "label"]]
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["validation"].to_pandas())
    test = proc_df(ds["test"].to_pandas())
    train_df = pd.concat([train, valid], ignore_index=True).reset_index(drop=True)
    train_df["idx"] = train_df.index
    test["idx"] = test.index
    train_df.to_csv(f"{data_dir}/base/newsmtc_train.csv", index=False)
    test.to_csv(f"{data_dir}/base/newsmtc_test.csv", index=False)
    return train_df, test

def process_imdb(random_seed=42, save=True):
    ds = load_dataset("stanfordnlp/imdb")
    train = ds["train"].to_pandas()
    test = ds["test"].to_pandas()
    train = train.sample(frac=0.25, random_state=random_seed)
    train["idx"] = train.index
    test["idx"] = test.index
    if save:
        train.to_csv(f"{data_dir}/base/imdb_train.csv", index=False)
        test.to_csv(f"{data_dir}/base/imdb_test.csv", index=False)
    return train, test

def process_financial_phrasebank(random_seed=42, save=True):
    ds = load_dataset("descartes100/enhanced-financial-phrasebank")
    train = ds["train"].to_pandas()
    train["text"] = train["train"].apply(lambda x: x["sentence"])
    train["label"] = train["train"].apply(lambda x: x["label"]) == 2
    train = train[["idx", "text", "label"]]
    train_df = train.sample(frac=0.75, random_state=random_seed)
    valid = train.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    train_df["idx"] = train_df.index
    valid["idx"] = valid.index
    if save:
        train_df.to_csv(f"{data_dir}/base/financial_phrasebank_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/financial_phrasebank_test.csv", index=False)
    return train_df, valid

def process_dair_emotion():
    ds = load_dataset("dair-ai/emotion", "split")
    train = ds["train"].to_pandas()
    valid = ds["validation"].to_pandas()
    test = ds["test"].to_pandas()
    def proc_df(df):
        df["label"] = df["label"].isin(['joy', 'love'])
        return df[["text", "label"]]
    train = proc_df(train)
    valid = proc_df(valid)
    test = proc_df(test)
    train_df = pd.concat([train, valid], ignore_index=True).reset_index(drop=True)
    train_df["idx"] = train_df.index
    test["idx"] = test.index
    train_df.to_csv(f"{data_dir}/base/dair_emotion_train.csv", index=False)
    test.to_csv(f"{data_dir}/base/dair_emotion_test.csv", index=False)
    return train_df, test
    
def process_colbert_humor(random_seed=42, save=True):
    ds = load_dataset("CreativeLang/ColBERT_Humor_Detection")
    train = ds["train"].to_pandas()
    train["label"] = train["humor"]
    train = train[["text", "label"]]
    train = train.sample(frac=0.15, random_state=random_seed).reset_index(drop=True) # 15% of 200k is 30k
    train_df = train.sample(frac=0.8, random_state=random_seed)
    valid = train.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    train_df["idx"] = train_df.index
    valid["idx"] = valid.index
    if save:
        train_df.to_csv(f"{data_dir}/base/colbert_humor_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/colbert_humor_test.csv", index=False)
    return train_df, valid

def process_epic_irony(random_seed=42, save=True):
    ds = load_dataset("CreativeLang/EPIC_Irony")
    train = ds["train"].to_pandas()
    train["label"] = train["label"] == "iro"
    train["initial_prompt"] = train["parent_text"]
    train["reply"] = train["text"]
    train["text"] = train["parent_text"] + "\n" +train["text"] 
    original_ids = train["id_original"].unique()
    train_ids = np.random.choice(original_ids, size=int(len(original_ids) * 0.75), replace=False)
    train = train[train["id_original"].isin(train_ids)].reset_index(drop=True)
    test = train[~train["id_original"].isin(train_ids)].reset_index(drop=True)
    def proc_df(df):
        scores = df.groupby("id_original")["label"].mean()
        df["label"] = df["id_original"].apply(lambda x: scores[x] > 0.5)
        df["idx"] = df.index
        return df[["idx", "text", "label", "initial_prompt", "reply"]]
    train = proc_df(train)
    test = proc_df(test)
    if save:
        train.to_csv(f"{data_dir}/base/epic_irony_train.csv", index=False)
        test.to_csv(f"{data_dir}/base/epic_irony_test.csv", index=False)
    return train, test

def process_sst5():
    ds = load_dataset("SetFit/sst5")
    train = ds["train"].to_pandas()
    valid = ds["validation"].to_pandas()
    test = ds["test"].to_pandas()
    def proc_df(df):
        df["label"] = df["label"] > 3
        return df[["text", "label"]]
    train = proc_df(train)
    valid = proc_df(valid)
    test = proc_df(test)
    train_df = pd.concat([train, valid], ignore_index=True).reset_index(drop=True)
    train_df["idx"] = train_df.index
    test["idx"] = test.index
    train_df.to_csv(f"{data_dir}/base/sst5_train.csv", index=False)
    test.to_csv(f"{data_dir}/base/sst5_test.csv", index=False)
    return train_df, test

def process_sms_spam():
    ds = load_dataset("ucirvine/sms_spam")
    train = ds["train"].to_pandas()
    train_df = train.sample(frac=0.8, random_state=42)
    valid = train.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    train_df["idx"] = train_df.index
    valid["idx"] = valid.index
    train_df.to_csv(f"{data_dir}/base/sms_spam_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/sms_spam_test.csv", index=False)
    return train_df, valid

def process_mops(random_state=42, save=True):
    ds = load_dataset("ManTle/mops", split="complete")
    df = ds.to_pandas()
    df = df[["premise", "theme"]]
    df["prompt_only"] = True
    train = df.sample(frac=0.8, random_state=random_state).reset_index(drop=True)
    valid = df.drop(train.index).reset_index(drop=True)
    train["idx"] = train.index
    valid["idx"] = valid.index
    if save:
        train.to_csv(f"{data_dir}/base/mops_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/mops_test.csv", index=False)
    test_themes = ["Time-travel"]
    train = df[~df["theme"].isin(test_themes)].reset_index(drop=True)
    test = df[df["theme"].isin(test_themes)].reset_index(drop=True)
    test["idx"] = test.index
    train.to_csv(f"{data_dir}/base/mops_domain_train.csv", index=False)
    test.to_csv(f"{data_dir}/base/mops_domain_test.csv", index=False)
    return train, valid



def save_dfs(train, valid, dataset_name, taskname):
    train.to_csv(f"{data_dir}/{taskname}/{dataset_name}_train.csv", index=False)
    valid.to_csv(f"{data_dir}/{taskname}/{dataset_name}_test.csv", index=False)

class ToxicityAvoidance:
    taskname = "toxicity_avoidance"

    def setup_toxic_chat(self, save=True):
        train = pd.read_csv(f"{data_dir}/base/toxic_chat_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/toxic_chat_test.csv")
        if save:
            save_dfs(train, valid, "toxic_chat", self.taskname)
        return train, valid
    
    def setup_real_toxicity_prompts(self, save=True):
        train = pd.read_csv(f"{data_dir}/base/real_toxicity_prompts_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/real_toxicity_prompts_test.csv")
        if save:
            save_dfs(train, valid, "real_toxicity_prompts", self.taskname)
        return train, valid


class Jailbreak:
    taskname = "jailbreak"
    def setup_toxic_chat(self, save=True):
        train = pd.read_csv(f"{data_dir}/base/toxic_chat_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/toxic_chat_test.csv")
        train["label"] = train["jailbreaking"]
        valid["label"] = valid["jailbreaking"]
        if save:
            save_dfs(train, valid, "toxic_chat", "jailbreak")
        return train, valid

class Unanswerable:
    taskname = "unanswerable"
    def setupsquad(self, save=True, random_seed=42):
        train = pd.read_csv(f"{data_dir}/base/squad_train.csv").sample(12_000, random_state=random_seed).reset_index(drop=True)
        valid = pd.read_csv(f"{data_dir}/base/squad_test.csv")
        def proc_df(df):
            df["text"] = "Answer the following question: \nQuestion: " + df["text"] + "\nAnswer: "
            df["label"] = df["unanswerable"].astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train, valid, "squad", self.taskname)
        return train, valid
    
    def setupqnota(self, save=True, random_seed=42):
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
                unanswerable = "Answer the following question: \nQuestion:" + row[file]['u']
                answerable = "Answer the following question: \nQuestion:" + row[file]['a']
                data.append([id_it, group_it, file, answerable, False])
                id_it += 1
                data.append([id_it, group_it, file, unanswerable, True])
                id_it += 1
                group_it += 1
        df = pd.DataFrame(data, columns=columns)
        train = df.sample(frac=0.8, random_state=random_seed)
        valid = df.drop(train.index).reset_index(drop=True)
        train = train.reset_index(drop=True)
        train["label"] = train["label"].astype(int)
        valid["label"] = valid["label"].astype(int)
        if save:
            save_dfs(train, valid, "qnota", self.taskname)
        return train, valid

    def setuphealthver(self, save=True):
        # will set up healthver for the NIE task
        train = pd.read_csv(f"{data_dir}/base/healthver_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/healthver_test.csv")
        def proc_df(df):
            df["text"] = "The following claim is either TRUE or FALSE. Which is it?\n" + df["text"] + "\nAnswer: "
            df["label"] = df["unanswerable"].astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train, valid, "healthver", self.taskname)
        return train, valid
    
    def setupselfaware(self, save=True):
        train = pd.read_csv(f"{data_dir}/base/selfaware_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/selfaware_test.csv")
        def proc_df(df):
            df["text"] = "Answer the following question\nQuestion: " + df["text"] + "\nAnswer: "
            df["label"] = df["unanswerable"].astype(int)
            wouldyous = df['text'].apply(lambda x: "would you rather" in x.lower())
            df = df[~wouldyous]
            df = df.reset_index(drop=True)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train, valid, "selfaware", self.taskname)
        return train, valid
    
    def setupknown_unknown(self, save=True):
        train = pd.read_csv(f"{data_dir}/base/known_unknown_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/known_unknown_test.csv")
        def proc_df(df):
            df["text"] = "Answer the following question\nQuestion: " + df["text"] + "\nAnswer: "
            df["label"] = df["unanswerable"].astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train, valid, "known_unknown", self.taskname)
        return train, valid

class NewsTopic:
    taskname = "newstopic"
    prompt_task_dict = {"summ": "Summarize the following news article: ", "answer": "Answer the question using information from the news article: "
                        , "question" : "Ask a question based on the news article: ", None: "", "topic": "What is the topic of the following news article? "}
    def setupagnews(self, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/agnews_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/agnews_test.csv")
        def proc_df(df):
            df["text"] = NewsTopic.prompt_task_dict[prompt_task] + df["text"] + "\n"
            keep_topics = ["Business", "Sci/Tech"]
            df = df[df["label_text"].isin(keep_topics)].reset_index(drop=True)
            df["label"] = df["label_text"] == "Sci/Tech"
            df = df[df["label_text"]]
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        prompt_task = "_"+prompt_task if prompt_task is not None else ""
        if save:
            save_dfs(train, valid, "agnews", self.taskname+prompt_task)
        return train, valid

    def setupbbcnews(self, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/bbcnews_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/bbcnews_test.csv")
        def proc_df(df):
            df["text"] = NewsTopic.prompt_task_dict[prompt_task] + df["text"] + "\n"
            keep_topics = ["business", "tech"]
            df = df[df["label_text"].isin(keep_topics)].reset_index(drop=True)
            df["label"] = df["label_text"] == "tech"
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        prompt_task = "_"+prompt_task if prompt_task is not None else ""
        if save:
            save_dfs(train, valid, "bbcnews", self.taskname+prompt_task)
        return train, valid
    
    def setupnytimes(self, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/nytimes_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/nytimes_test.csv")
        def proc_df(df):
            df["text"] = NewsTopic.prompt_task_dict[prompt_task] + df["text"] + "\n"
            keep_topics = ["Science", "Real Estate", "Economy", "Technology", "Your Money", "Global Business"]
            df = df[df["label_text"].isin(keep_topics)].reset_index(drop=True)
            df["label"] = df["label_text"].isin(["Technology", "Science"])
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        prompt_task = "_"+prompt_task if prompt_task is not None else ""
        if save:
            save_dfs(train, valid, "nytimes", self.taskname+prompt_task)
        return train, valid

class Sentiment:
    taskname = "sentiment"
    prompt_task_dict = {"speaker": "Speaker 1: ", "question": "Ask a question based on the following prompt: ", "answer": "Answer the following question: ", None: "", "sentiment": "What is the sentiment of the following text? "}

    def setupstandard(self, name, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/{name}_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/{name}_test.csv")
        def proc_df(df):
            df["text"] = Sentiment.prompt_task_dict[prompt_task] + df["text"]
            df["label"] = df["label"].astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        prompt_task = "_"+prompt_task if prompt_task is not None else ""
        if save:
            save_dfs(train, valid, name, self.taskname+prompt_task)
        return train, valid
    
    def setupamazonreviews(self, save=True, prompt_task=None):
        return self.setupstandard("amazonreviews", save, prompt_task)
    
    def setupyelp(self, save=True, prompt_task=None):
        return self.setupstandard("yelp", save, prompt_task)
    
    def setuptwitterfinance(self, save=True, prompt_task=None):
        return self.setupstandard("twitterfinance", save, prompt_task)
    
    def setuptwittermteb(self, save=True, prompt_task=None):
        return self.setupstandard("twittermteb", save, prompt_task)
    
    def setupauditorsentiment(self, save=True, prompt_task=None):
        return self.setupstandard("auditorsentiment", save, prompt_task)
    
    def setupfiqa(self, save=True, prompt_task=None):
        return self.setupstandard("fiqa", save, prompt_task)
    
    def setupindosentiment(self, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/indosentiment_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/indosentiment_test.csv")
        def proc_df(df, text_col="text"):
            df = df[["idx", "text", "label"]]
            df["text"] = Sentiment.prompt_task_dict[prompt_task] + df["text"]
            df["label"] = df["label"].astype(int)
            return df
        train_df = proc_df(train)
        valid_df = proc_df(valid)
        prompt_task = "_"+prompt_task if prompt_task is not None else ""
        if save:
            save_dfs(train_df, valid_df, "indosentiment_eng", self.taskname+prompt_task)
        train_df = proc_df(train, "alt_text")
        valid_df = proc_df(valid, "alt_text")
        if save:
            save_dfs(train_df, valid_df, "indosentiment_ind", self.taskname+prompt_task)
        return train, valid

    def setupnewsmtc(self, save=True, prompt_task=None):
        return self.setupstandard("newsmtc", save, prompt_task)

    def setupimdb(self, save=True, prompt_task=None):
        return self.setupstandard("imdb", save, prompt_task)

    def setupfinancial_phrasebank(self, save=True, prompt_task=None):
        return self.setupstandard("financial_phrasebank", save, prompt_task)

    def setupdair_emotion(self, save=True, prompt_task=None):
        return self.setupstandard("dair_emotion", save, prompt_task)

    def setup_sst5(self, save=True, prompt_task=None):
        return self.setupstandard("sst5", save, prompt_task)


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
    taskname = "confidence"

    def setupmmlu(self, k=2, save=True, random_seed=42):
        train = pd.read_csv(f"{data_dir}/base/mmlu_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/mmlu_test.csv")
        system_prompt = "Answer the following MCQ by providing the correct option"
        def proc_df(df):
            df["choices"] = df["choices"].apply(eval)
            for i in range(len(df)):
                subject = df.loc[i, "subject"]
                train_subjects = train[(train["subject"] == subject) & (train["idx"] != df.loc[i, "idx"])].reset_index(drop=True)
                train_subjects = train_subjects.sample(k, random_state=random_seed).reset_index(drop=True)
                prompt = ""
                for j in range(k):
                    prompt += f"\nQuestion: {train_subjects.loc[j, 'question']}{mmlu_choices_to_text(train_subjects.loc[j, 'choices'])}\nAnswer: {mmlu_answer_to_letter(train_subjects.loc[j, 'answer'])} [STOP]"
                df.loc[i, "text"] = f"{system_prompt}\n{prompt}\nQuestion: {df.loc[i, 'question']}{mmlu_choices_to_text(df.loc[i, 'choices'])}\nAnswer: "
                df.loc[i, "gold"] = mmlu_answer_to_letter(df.loc[i, "answer"])
            return df[["idx", "text", "gold"]]
        train_df = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train_df, valid, "mmlu", self.taskname)
        return train_df, valid

def process_all():
    process_squad()
    process_healthver()
    process_selfaware()
    process_known_unkown()
    process_mmlu()
    process_real_toxicity_prompts()
    process_toxic_chat()
    process_qnota()
    process_agnews()
    process_bbcnews()
    process_nytimes()
    process_amazonreviews()
    process_yelp()
    process_twitterfinance()
    process_twittermteb()
    process_auditor()
    process_fiqa()
    process_indosentiment()
    process_newsmtc()
    process_imdb()
    process_financial_phrasebank()
    process_dair_emotion()
    process_colbert_humor()
    process_epic_irony()
    process_sst5()
    process_sms_spam()
    process_mops()
    return


def setup_all():
    unanswerable = Unanswerable()
    toxicity = ToxicityAvoidance()
    jailbreak = Jailbreak()
    confidence = Confidence()
    news_topic = NewsTopic()
    sentiment = Sentiment()
    unanswerable.setuphealthver()
    unanswerable.setupqnota()
    unanswerable.setupselfaware()
    unanswerable.setupsquad()
    unanswerable.setupknown_unknown()
    confidence.setupmmlu()
    toxicity.setup_real_toxicity_prompts()
    toxicity.setup_toxic_chat()
    jailbreak.setup_toxic_chat()
    for key in news_topic.prompt_task_dict:
        news_topic.setupagnews(prompt_task=key)
        news_topic.setupbbcnews(prompt_task=key)
        news_topic.setupnytimes(prompt_task=key)
    for key in sentiment.prompt_task_dict:
        sentiment.setupamazonreviews(prompt_task=key)
        sentiment.setupyelp(prompt_task=key)
        sentiment.setuptwitterfinance(prompt_task=key)
        sentiment.setuptwittermteb(prompt_task=key)
        sentiment.setupauditorsentiment(prompt_task=key)
        sentiment.setupfiqa(prompt_task=key)
        sentiment.setupindosentiment(prompt_task=key)
        sentiment.setupnewsmtc(prompt_task=key)
        sentiment.setupimdb(prompt_task=key)
        sentiment.setupfinancial_phrasebank(prompt_task=key)
        sentiment.setupdair_emotion(prompt_task=key)
        sentiment.setup_sst5(prompt_task=key)
    return



if __name__ == "__main__":
    process_all()
    setup_all()