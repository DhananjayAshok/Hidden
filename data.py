import pandas as pd
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
import re
import warnings

def remove_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Substitute URLs with an empty string
    cleaned_text = url_pattern.sub('', text)
    return cleaned_text

data_dir = os.environ["DATA_DIR"]
if not os.path.exists(data_dir):
    raise ValueError(f"DATA_DIR {data_dir} does not exist. Please set the DATA_DIR environment variable to the directory where you want to save the data and then run the scripts/get_data.sh script.")
if not os.path.exists(data_dir+"/base/"):
    os.makedirs(data_dir+"/base/")

maximum_train_size = 5000 # Will never save more than this number of training examples
global_random_seed = 42
np.random.seed(global_random_seed)

def save_dfs(train, valid, dataset_name, taskname, prompt_task=None):
    if prompt_task is None:
        prompt_task = ""
    else:
        prompt_task = "_"+prompt_task
    if not os.path.exists(f"{data_dir}/{taskname}/"):
        os.makedirs(f"{data_dir}/{taskname}/")
    if maximum_train_size is not None:
        if len(train) > maximum_train_size:
            train = train.sample(n=maximum_train_size, random_state=global_random_seed).reset_index(drop=True)
    train.to_csv(f"{data_dir}/{taskname}{prompt_task}/{dataset_name}_train.csv", index=False)
    valid.to_csv(f"{data_dir}/{taskname}{prompt_task}/{dataset_name}_test.csv", index=False)


# News Topic
#region

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

#endregion

# Unanswerable, Fact Verification and Truth
#region

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

def process_climate_fever(random_seed=42, save=True):
    ds = load_dataset("tdiggelm/climate_fever")
    df = ds["test"].to_pandas()
    def accumilate_evidence(evidence_list):
        complete_evidence = ""
        for evidence in evidence_list:
            complete_evidence = complete_evidence + " " + evidence["evidence"]
        return complete_evidence
    df["evidence"] = df["evidences"].apply(accumilate_evidence)
    df["text"] = "Evidence: " + df["evidence"] + "\nClaim: " + df["claim"]
    df["label"] = df["claim_label"]
    df["unanswerable"] = df["label"] == 2
    train = df.sample(frac=0.8, random_state=random_seed)
    valid = df.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    train["idx"] = train.index
    valid["idx"] = valid.index
    if save:
        train.to_csv(f"{data_dir}/base/climatefever_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/climatefever_test.csv", index=False)

def process_factool(random_seed=42, save=True):
    def proc_df(df):
        data = []
        columns = ["claim", "label"]
        for i, row in df.iterrows():
            claims_w_label = row["claims"]
            for claim_w_label in claims_w_label:
                data.append([claim_w_label["claim"], claim_w_label["label"]])
        df = pd.DataFrame(data, columns=columns)
        df["text"] = df["claim"]
        df["idx"] = df.index
        return df
    total = pd.read_json(f"{data_dir}/raw/factool/knowledge_qa.jsonl", lines=True)
    train = total.sample(frac=0.2, random_state=random_seed)
    valid = total.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    train = proc_df(train)
    valid = proc_df(valid)
    if save:
        train.to_csv(f"{data_dir}/base/factool_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/factool_test.csv", index=False)
    return train, valid

def process_felm(random_seed=42, save=True):
    train_dfs = []
    valid_dfs = []
    for subset in ["wk", "science"]:
        ds = load_dataset("hkust-nlp/felm", subset)
        def proc_df(df):
            df["text"] = df["prompt"] + " "+ df["response"]
            return df
        train = proc_df(ds["test"].to_pandas())
        train_df = train.sample(frac=0.2, random_state=random_seed)
        valid = train.drop(train_df.index).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        train_df["idx"] = train_df.index
        valid["idx"] = valid.index
        train_dfs.append(train_df)
        valid_dfs.append(valid)
    train_df = pd.concat(train_dfs, ignore_index=True)
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    if save:
        train_df.to_csv(f"{data_dir}/base/felm_train.csv", index=False)
        valid_df.to_csv(f"{data_dir}/base/felm_test.csv", index=False)
    return train_df, valid_df

def process_averitec():
    ds = load_dataset("pminervini/averitec")
    def proc_df(df):
        df["idx"] = df.index
        return df
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["dev"].to_pandas())
    train.to_csv(f"{data_dir}/base/averitec_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/averitec_test.csv", index=False)
    return train, valid

def process_fever(random_seed=42, save=True):
    ds = load_dataset("fever/fever", 'v1.0', trust_remote_code=True)
    train = ds["train"].to_pandas()
    valid = ds["labelled_dev"].to_pandas()
    def proc_df(df):
        df = df.drop_duplicates(subset=["claim"]).reset_index(drop=True)
        return df
    train = proc_df(train)
    valid = proc_df(valid)
    train = train.sample(frac=0.5, random_state=random_seed)
    valid = valid.sample(frac=0.6, random_state=random_seed)
    train["idx"] = train.index
    valid["idx"] = valid.index
    if save:
        train.to_csv(f"{data_dir}/base/fever_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/fever_test.csv", index=False)
    return train, valid

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

def process_qnota():
    # ambiguous, futuristic, unmeasurable, incorrect
    columns = ["incomplete_questions", "ambiguous_questions", "futuristic_questions", "unmeasurable_questions", "incorrect_questions"]
#endregion

# Sentiment and Tone
#region
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
    valid = proc_df(ds["eval"].to_pandas())
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
        df["text"] = df["sentence"]
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
        return df[["text", "label"]]
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
        return df[["text", "label"]]
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
    train = train[["text", "label"]]
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

def process_mops(random_seed=42, save=True):
    ds = load_dataset("ManTle/mops", split="complete")
    df = ds.to_pandas()
    df = df[["premise", "theme"]]
    df["prompt_only"] = True
    train = df.sample(frac=0.8, random_state=random_seed).reset_index(drop=True)
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

#endregion

# MCQA
#region

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


def process_cosmoqa():
    ds = load_dataset("allenai/cosmos_qa", trust_remote_code=True)
    def proc_df(df):
        df["choices"] = df["answer0"].apply(lambda x: [x]) + df["answer1"].apply(lambda x: [x]) + df["answer2"].apply(lambda x: [x]) + df["answer3"].apply(lambda x: [x])
        df["question_save"] = df["question"]
        df["question"] = "Context: " + df["context"] + "\nQuestion: " + df["question"]
        df["answer"] = df["label"]
        return df
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["validation"].to_pandas())
    test = proc_df(ds["test"].to_pandas())
    train_df = pd.concat([train, valid], ignore_index=True).reset_index(drop=True)
    train_df["idx"] = train_df.index
    test["idx"] = test.index
    train_df.to_csv(f"{data_dir}/base/cosmoqa_train.csv", index=False)
    test.to_csv(f"{data_dir}/base/cosmoqa_test.csv", index=False)
    return train_df, test

def process_piqa():
    ds = load_dataset("ybisk/piqa", trust_remote_code=True)
    def proc_df(df):
        df["idx"] = df.index
        df["choices"] = df["sol1"].apply(lambda x: [x]) + df["sol2"].apply(lambda x: [x])
        df["answer"] = df["label"].astype(int)
        nan_cols = df[["goal", "choices", "answer"]].isna().any(axis=1)
        df = df[~nan_cols].reset_index(drop=True)
        return df
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["validation"].to_pandas())
    train.to_csv(f"{data_dir}/base/piqa_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/piqa_test.csv", index=False)

def process_arc():
    train_dfs = []
    valid_dfs = []
    for subset in ["ARC-Easy", "ARC-Challenge"]:
        ds = load_dataset("ai2_arc", subset)
        def proc_df(df):
            df = df[df["choices"].apply(lambda x: len(x["label"]) == 4)].reset_index(drop=True)
            answer_keymap = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}
            df["answer"] = df["answerKey"].map(answer_keymap)
            df['choices'] = df['choices'].apply(lambda x: x['text'].tolist())
            df["challenge"] = (subset == "ARC-Challenge")
            return df
        train = proc_df(ds["train"].to_pandas())
        valid = proc_df(ds["validation"].to_pandas())
        test = proc_df(ds["test"].to_pandas())
        train_df = pd.concat([train, valid], ignore_index=True).reset_index(drop=True)
        train_dfs.append(train_df)
        valid_dfs.append(test)
    train_df = pd.concat(train_dfs, ignore_index=True).reset_index(drop=True)
    valid_df = pd.concat(valid_dfs, ignore_index=True).reset_index(drop=True)
    train_df["idx"] = train_df.index
    valid_df["idx"] = valid_df.index
    train_df.to_csv(f"{data_dir}/base/arc_train.csv", index=False)
    valid_df.to_csv(f"{data_dir}/base/arc_test.csv", index=False)
    return train, valid

def process_medmcqa(random_seed=42, save=True):
    ds = load_dataset("openlifescienceai/medmcqa")
    def proc_df(df):
        df["choices"] = df["opa"].apply(lambda x: [x]) + df["opb"].apply(lambda x: [x]) + df["opc"].apply(lambda x: [x]) + df["opd"].apply(lambda x: [x])
        df["answer"] = df["cop"]
        return df[["question", "choices", "answer"]]
    train = ds["train"].to_pandas().sample(frac=0.1, random_state=random_seed).reset_index(drop=True)
    train = proc_df(train)
    valid = proc_df(ds["validation"].to_pandas())
    test = proc_df(ds["test"].to_pandas())
    train_df = pd.concat([train, valid], ignore_index=True).reset_index(drop=True)
    train_df["idx"] = train_df.index
    test["idx"] = test.index
    if save:
        train_df.to_csv(f"{data_dir}/base/medmcqa_train.csv", index=False)
        test.to_csv(f"{data_dir}/base/medmcqa_test.csv", index=False)
    return train_df, test

def process_commonsenseqa():
    ds = load_dataset("tau/commonsense_qa")
    def proc_df(df):
        df["choices"] = df["choices"].apply(lambda x: x["text"].tolist())
        df["answer"] = df["answerKey"]
        df["idx"] = df.index
        return df[["idx", "question", "choices", "answer"]]
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["validation"].to_pandas())
    train.to_csv(f"{data_dir}/base/commonsenseqa_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/commonsenseqa_test.csv", index=False)
    return train, valid

def process_openbookqa():
    ds = load_dataset("allenai/openbookqa", "main")
    def proc_df(df):
        df["choices"] = df["choices"].apply(lambda x: x["text"].tolist())
        df["answer"] = df["answerKey"]
        df["question"] = df["question_stem"]
        return df[["question", "choices", "answer"]]
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["validation"].to_pandas())
    test = proc_df(ds["test"].to_pandas())
    train_df = pd.concat([train, valid], ignore_index=True).reset_index(drop=True)
    train_df["idx"] = train_df.index
    test["idx"] = test.index
    train_df.to_csv(f"{data_dir}/base/openbookqa_train.csv", index=False)
    test.to_csv(f"{data_dir}/base/openbookqa_test.csv", index=False)
    return train_df, test

def process_qasc():
    ds = load_dataset("allenai/qasc")
    def proc_df(df):
        df["choices"] = df["choices"].apply(lambda x: x["text"].tolist())
        df["answer"] = df["answerKey"]
        df["idx"] = df.index
        return df[["idx", "question", "choices", "answer"]]
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["validation"].to_pandas())
    train.to_csv(f"{data_dir}/base/qasc_train.csv", index=False)
    valid.to_csv(f"{data_dir}/base/qasc_test.csv", index=False)
    return train, valid
    
def process_hellaswag(random_seed=42, save=True):
    ds = load_dataset("AlekseyKorshuk/hellaswag")
    def proc_df(df):
        df["choices"] = df["endings"].apply(lambda x: x.tolist())
        df["text"] = df["ctx"]
        df["answer"] = df["label"]
        return df[["text", "choices", "answer"]]
    train = proc_df(ds["train"].to_pandas())
    valid = proc_df(ds["validation"].to_pandas())
    test = proc_df(ds["test"].to_pandas())
    train_df = pd.concat([train, valid], ignore_index=True).sample(frac=0.35, random_state=random_seed).reset_index(drop=True)
    train_df["idx"] = train_df.index
    test["idx"] = test.index
    if save:
        train_df.to_csv(f"{data_dir}/base/hellaswag_train.csv", index=False)
        test.to_csv(f"{data_dir}/base/hellaswag_test.csv", index=False)

def process_bigbenchhard(random_seed=42, save=True):
    subsets = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies']
    # mcq is date_und, disamb, geometric, hyperbaton, logicals, movie_reco, penguins, reasoning, ruin, salient, snarks, temporal, shuffled
    mcqs = ["date_understanding", "disambiguation_qa", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation", "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection", "snarks", "temporal_sequences", "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects"]
    numerical = ["multistep_arithmetic_two", "object_counting"]
    # remaining are binary
    binary = [x for x in subsets if x not in mcqs and x not in numerical]
    train_dfs = {"mcq": [], "numerical": [], "all": []}
    test_dfs = {"mcq": [], "numerical": [],  "all": []}

    def get_choices(choice_str):
        # look for the regex pattern (LETTER) OPTION TEXT and split by (LETTER)
        options = re.split(r"\([A-Z]\)", choice_str)[1:]
        return [x.strip() for x in options]

    for subset in subsets:
        ds = load_dataset("maveriq/bigbenchhard", subset) # there are exactly 250 examples per subset, 200 for test rest train
        df = ds["train"].to_pandas()
        if subset == "movie_recommendation":
            df = df[df["target"] != "Monsters, Inc"].reset_index(drop=True)
        if subset == "ruin_names":
            df = df[df["target"] != "dearth, wind, & fire"].reset_index(drop=True)
            df = df[df["target"] != "rita, sue and bob poo"].reset_index(drop=True)
        df["text"] = df["input"]
        if isinstance(df["target"][0], str):
            df["answer"] = df["target"].apply(lambda x: x.strip("()"))
        else:
            df["answer"] = df["target"]
        df["subset"] = subset
        if subset in mcqs or subset in binary:
            if "Options:" in df["text"][0]:
                df["question"] = df["text"].apply(lambda x: x.split("Options:")[0])
            else:
                df["question"] = df["text"]
            if subset in binary:
                all_options = df["answer"].unique().tolist()
                df["choices"] = str(all_options)
                df["answer"] = df["target"].apply(lambda x: all_options.index(x))
            else:
                df["choices"] = df["text"].apply(lambda x: get_choices(x.split("Options:")[1]))
                df["answer"] = df["answer"].apply(letter_to_int)
        train_df = df.sample(n=50, random_state=random_seed)
        test_df = df.drop(train_df.index).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        train_df["idx"] = train_df.index
        test_df["idx"] = test_df.index
        if subset in mcqs or subset in binary:
            train_dfs["mcq"].append(train_df)
            test_dfs["mcq"].append(test_df)
        elif subset in numerical:
            assert train_df["answer"].nunique() > 10
            train_dfs["numerical"].append(train_df)
            test_dfs["numerical"].append(test_df)
        else:
            assert train_df["answer"].nunique() == 2
            train_dfs["binary"].append(train_df)
            test_dfs["binary"].append(test_df)
        train_dfs["all"].append(train_df)
        test_dfs["all"].append(test_df)
    for kind in train_dfs:
        train_df = pd.concat(train_dfs[kind], ignore_index=True)
        test_df = pd.concat(test_dfs[kind], ignore_index=True)
        if save:
            train_df.to_csv(f"{data_dir}/base/bigbenchhard_{kind}_train.csv", index=False)
            test_df.to_csv(f"{data_dir}/base/bigbenchhard_{kind}_test.csv", index=False)
    return train_df, test_df

def process_truthfulqa(random_seed=42, save=True):
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    train = ds["validation"].to_pandas()
    train["choices"] = train["mc1_targets"].apply(lambda x: x["choices"].tolist())
    train["answer"] = train["mc1_targets"].apply(lambda x: x["labels"].tolist().index(1))
    train_df = train.sample(frac=0.2, random_state=random_seed)
    valid = train.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    train_df["idx"] = train_df.index
    valid["idx"] = valid.index
    if save:
        train_df.to_csv(f"{data_dir}/base/truthfulqa_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/truthfulqa_test.csv", index=False)
    ds = load_dataset("truthfulqa/truthful_qa", "generation")
    df = ds["validation"].to_pandas()
    def proc_df(df):
        data = []
        columns = ["question", "claim", "label"]
        for i, row in df.iterrows():
            question = row["question"]
            correct_claims = row["correct_answers"]
            incorrect_claims = row["incorrect_answers"]
            for claim in correct_claims:
                data.append([question, claim, 1])
            for claim in incorrect_claims:
                data.append([question, claim, 0])
        df = pd.DataFrame(data, columns=columns)
        df["idx"] = df.index
        return df
    train = df.sample(frac=0.2, random_state=random_seed)
    valid = df.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    train = proc_df(train)
    valid = proc_df(valid)
    if save:
        train.to_csv(f"{data_dir}/base/truthfulqa_gen_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/truthfulqa_gen_test.csv", index=False)
    return train, valid  

#endregion


# Hallucination
#region
def process_ragtruth(random_seed=42, save=True):
    df = pd.read_json(f"{data_dir}/raw/ragtruth/source_info.jsonl", lines=True)
    def proc_df(df):
        df["text"] = df["prompt"]
        df["idx"] = df.index
        return df
    train = df.sample(frac=0.85, random_state=random_seed)
    valid = df.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    train = proc_df(train)
    valid = proc_df(valid)
    if save:
        train.to_csv(f"{data_dir}/base/ragtruth_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/ragtruth_test.csv", index=False)
    return train, valid


def process_faithbench(random_seed=42, save=True):
    df = pd.read_csv(f"{data_dir}/raw/faithbench/FaithBench.csv")
    df = df[["source"]]
    df = df.drop_duplicates().reset_index(drop=True)
    df.rename(columns={"source": "text"}, inplace=True)
    train = df.sample(frac=0.25, random_state=random_seed)
    valid = df.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    train["idx"] = train.index
    valid["idx"] = valid.index
    if save:
        train.to_csv(f"{data_dir}/base/faithbench_train.csv", index=False)
        valid.to_csv(f"{data_dir}/base/faithbench_test.csv", index=False)
    return train, valid

#endregion



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
    
    def setupclimatefever(self, save=True):
        train = pd.read_csv(f"{data_dir}/base/climatefever_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/climatefever_test.csv")
        def proc_df(df):
            df["text"] = "The following claim is either TRUE or FALSE. Which is it?\n" + df["text"] + "\nAnswer: "
            df["label"] = df["unanswerable"].astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train, valid, "climatefever", self.taskname)
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
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        prompt_task = "_"+prompt_task if prompt_task is not None else ""
        if save:
            save_dfs(train, valid, "agnews", self.taskname, prompt_task)
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
            save_dfs(train, valid, "bbcnews", self.taskname, prompt_task)
        return train, valid
    
    def setupnytimes(self, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/nytimes_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/nytimes_test.csv")
        def proc_df(df):
            df["text"] = NewsTopic.prompt_task_dict[prompt_task] + df["text"] + "\n"
            keep_topics = ["Science", "Real Estate", "Economy", "Technology", "Your Money", "Global Business"]
            df = df[df["section"].isin(keep_topics)].reset_index(drop=True)
            df["label"] = df["section"].isin(["Technology", "Science"])
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        prompt_task = "_"+prompt_task if prompt_task is not None else ""
        if save:
            save_dfs(train, valid, "nytimes", self.taskname, prompt_task)
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
            save_dfs(train, valid, name, self.taskname, prompt_task)
        return train, valid
    
    def setup_amazonreviews(self, save=True, prompt_task=None):
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
        train = pd.read_csv(f"{data_dir}/base/indosentiment_train.csv", lineterminator="\n")
        valid = pd.read_csv(f"{data_dir}/base/indosentiment_test.csv", lineterminator="\n")
        def proc_df(df, text_col="text"):
            df = df[["idx", "text", "label"]]
            df["text"] = Sentiment.prompt_task_dict[prompt_task] + df["text"]
            df["label"] = df["label"].astype(int)
            return df
        train_df = proc_df(train)
        valid_df = proc_df(valid)
        if save:
            save_dfs(train_df, valid_df, "indosentiment_eng", self.taskname, prompt_task)
        train_df = proc_df(train, "alt_text")
        valid_df = proc_df(valid, "alt_text")
        if save:
            save_dfs(train_df, valid_df, "indosentiment_ind", self.taskname, prompt_task)
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


def choices_to_text(choices):
    text = ""
    for i, choice in enumerate(choices):
        text += f"\nOption {int_to_letter(i)}: {choice}"
    return text

def int_to_letter(answer):
    # Return the letter corresponding to the integer answer in capital
    if isinstance(answer, str):
        assert answer in "ABCDEFGHIJKLMNOPQRST"
        return answer
    assert answer in range(20)
    return chr(answer + 65)

def letter_to_int(answer):
    # Return the integer corresponding to the letter answer
    if isinstance(answer, int):
        assert answer in range(20)
        return answer
    assert answer in "ABCDEFGHIJKLMNOPQRST"
    return ord(answer) - 65


def randomize_choices(choices, answer, force_total=None):
    # Randomize the choices and return the new choices and the new answer
    remember_dtype = "int" if isinstance(answer, int) else "str"
    if isinstance(answer, str):
        answer = letter_to_int(answer)
    answer_text = choices[answer]
    for choice in choices:
        if choices.count(choice) > 1:
            #print(f"Warning: {choice} is repeated in {choices}")
            pass
    if force_total is not None:
        if len(choices) < force_total:
            warnings.warn(f"Warning: {len(choices)} choices is less than {force_total}")
        remaining = [x for x in choices if x != answer_text]
        np.random.shuffle(remaining)
        choices = [answer_text]
        choices += remaining[:force_total-1]
    np.random.shuffle(choices)
    new_answer = choices.index(answer_text)
    if remember_dtype == "str":
        new_answer = int_to_letter(new_answer)
    return choices, new_answer


class Confidence:
    taskname = "confidence"
    system_prompt = "Answer the following MCQ by providing the correct option"

    def setupstandard(self, name, question_column="question", subset_col=None, save=True, random_seed=42, k=2, force_total=None, train=None, valid=None):
        if train is None or valid is None:
            train = pd.read_csv(f"{data_dir}/base/{name}_train.csv")
            valid = pd.read_csv(f"{data_dir}/base/{name}_test.csv")
        def proc_df(df):
            nan_cols = df[["idx", question_column, "choices", "answer"]].isna().any(axis=1)
            df = df[~nan_cols].reset_index(drop=True)
            if not isinstance(df.loc[0, "choices"], list):
                df["choices"] = df["choices"].apply(eval)
            if not isinstance(df.loc[0, "answer"], str):
                df["answer"] = df["answer"].astype(int)
            for i in tqdm(range(len(df))):
                prompt_candidates = df[df["idx"] != df.loc[i, "idx"]].reset_index(drop=True)
                if subset_col is not None:
                    subset = df.loc[i, subset_col]
                    prompt_candidates = prompt_candidates[prompt_candidates[subset_col] == subset]
                prompt_index = np.random.choice(prompt_candidates.index, k, replace=False)
                prompt_selected = prompt_candidates.loc[prompt_index].reset_index(drop=True)
                prompt = Confidence.system_prompt
                for j in range(k):
                    question_component = f"\nQuestion: {prompt_selected.loc[j, question_column]}"
                    choices, answer = randomize_choices(prompt_selected.loc[j, "choices"], prompt_selected.loc[j, "answer"], force_total=force_total)
                    choices_component = choices_to_text(choices)
                    answer_component = f"\nAnswer: {int_to_letter(answer)} [STOP]"
                    prompt = prompt + question_component + choices_component + answer_component
                own_question_component = f"\nQuestion: {df.loc[i, question_column]}"
                choices, answer = randomize_choices(df.loc[i, "choices"], df.loc[i, "answer"], force_total=force_total)
                own_choices_component = choices_to_text(choices)
                df.loc[i, "text"] = prompt + own_question_component + own_choices_component + "\nAnswer: "
                df.loc[i, "gold"] = int_to_letter(answer)
            return df
        train = proc_df(train)
        valid = proc_df(valid)
        savename = ""
        if force_total is not None:
            savename += f"_{force_total}_ops"
        if save:
            save_dfs(train, valid, name, self.taskname+savename)
        return train, valid

    def setup_mmlu(self, k=2, save=True, random_seed=42):
        train, valid = self.setupstandard("mmlu", question_column="question", subset_col="subject", save=save, random_seed=random_seed, k=k)
        train, valid = self.setupstandard("mmlu", question_column="question", subset_col="subject", save=save, random_seed=random_seed, k=k, force_total=2)
        return train, valid
    
    def setup_cosmoqa(self, k=2, save=True, random_seed=42):
        train, valid = self.setupstandard("cosmoqa", question_column="question", save=save, random_seed=random_seed, k=k)
        train, valid = self.setupstandard("cosmoqa", question_column="question", save=save, random_seed=random_seed, k=k, force_total=2)
        return train, valid
    
    def setup_piqa(self, k=2, save=True, random_seed=42):
        # TODO: Eval fails here
        train, valid = self.setupstandard("piqa", question_column="goal", save=save, random_seed=random_seed, k=k)
        train, valid = self.setupstandard("piqa", question_column="goal", save=save, random_seed=random_seed, k=k, force_total=2)
        return train, valid
    
    def setup_arc(self, k=2, save=True, random_seed=42):
        train, valid = self.setupstandard("arc", question_column="question", save=save, random_seed=random_seed, k=k)
        train, valid = self.setupstandard("arc", question_column="question", save=save, random_seed=random_seed, k=k, force_total=2)
        return train, valid
    
    def setup_medmcqa(self, k=2, save=True, random_seed=42):
        train, valid = self.setupstandard("medmcqa", question_column="question", save=save, random_seed=random_seed, k=k)
        train, valid = self.setupstandard("medmcqa", question_column="question", save=save, random_seed=random_seed, k=k, force_total=2)
        return train, valid
    
    def setup_commonsenseqa(self, k=2, save=True, random_seed=42):
        train, valid = self.setupstandard("commonsenseqa", question_column="question", save=save, random_seed=random_seed, k=k)
        train, valid = self.setupstandard("commonsenseqa", question_column="question", save=save, random_seed=random_seed, k=k, force_total=2)
        return train, valid
    
    def setup_openbookqa(self, k=2, save=True, random_seed=42):
        train, valid = self.setupstandard("openbookqa", question_column="question", save=save, random_seed=random_seed, k=k)
        train, valid = self.setupstandard("openbookqa", question_column="question", save=save, random_seed=random_seed, k=k, force_total=2)
        return train, valid
    
    def setup_qasc(self, k=2, save=True, random_seed=42):
        train, valid = self.setupstandard("qasc", question_column="question", save=save, random_seed=random_seed, k=k)
        train, valid = self.setupstandard("qasc", question_column="question", save=save, random_seed=random_seed, k=k, force_total=2)
        return train, valid
    
    def setup_hellaswag(self, k=2, save=True, random_seed=42):
        train = pd.read_csv(f"{data_dir}/base/hellaswag_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/hellaswag_test.csv")
        train["question"] = "Which of the following continuations to the text are the most appropriate? \nText: " + train["text"]
        valid["question"] = "Which of the following continuations to the text are the most appropriate? \nText: " + valid["text"]
        self.setupstandard("hellaswag", question_column="text", save=save, random_seed=random_seed, k=k)
        train, valid = self.setupstandard("hellaswag", question_column="text", save=save, random_seed=random_seed, k=k, force_total=2)
        return train, valid
    
    def setup_bigbenchhard(self, k=2, save=True, random_seed=42):
        train, valid = self.setupstandard("bigbenchhard_mcq", question_column="text", save=save, random_seed=random_seed, k=k)
        train, valid = self.setupstandard("bigbenchhard_mcq", question_column="text", save=save, random_seed=random_seed, k=k, force_total=2)
        return train, valid

    def setup_truthfulqa(self, k=2, save=True, random_seed=42):
        train, valid = self.setupstandard("truthfulqa", question_column="question", save=save, random_seed=random_seed, k=k)
        train, valid = self.setupstandard("truthfulqa", question_column="question", save=save, random_seed=random_seed, k=k, force_total=2)
        return train, valid


class Truthfullness:
    taskname = "truthfullness"
    prompt_task_dict = {"speaker": "Speaker 1: ", None: "", "truth": "Is the following claim true?: "}
    def setup_felm(self, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/felm_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/felm_test.csv")
        def proc_df(df):
            df["text"] = Truthfullness.prompt_task_dict[prompt_task] + df["text"]
            df['label'] = df['labels'].apply(all).astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train, valid, "felm", self.taskname, prompt_task)
        return train, valid
    
    def setup_healthver(self, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/healthver_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/healthver_test.csv")
        def proc_df(df):
            df["text"] = Truthfullness.prompt_task_dict[prompt_task] + df["claim"]
            df['label'] = (df['label'] == "Supports").astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train, valid, "healthver", self.taskname, prompt_task)
        return train, valid
    
    def setup_climate_fever(self, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/climatefever_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/climatefever_test.csv")
        def proc_df(df):
            df["text"] = Truthfullness.prompt_task_dict[prompt_task] + df["claim"]
            df['label'] = (df['label'] == 0).astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train, valid, "climatefever", self.taskname, prompt_task)
        return train, valid
    
    def setup_averitec(self, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/averitec_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/averitec_test.csv")
        def proc_df(df):
            df["text"] = Truthfullness.prompt_task_dict[prompt_task] + df["claim"]
            df['label'] = (df['label'] == "Supported").astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train, valid, "averitec", self.taskname, prompt_task)
        return train, valid
    
    def setup_fever(self, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/fever_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/fever_test.csv")
        def proc_df(df):
            df["text"] = Truthfullness.prompt_task_dict[prompt_task] + df["claim"]
            df['label'] = (df['label'] == "SUPPORTS").astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train, valid, "fever", self.taskname, prompt_task)
        return train, valid
    
    def setup_factool(self, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/factool_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/factool_test.csv")
        def proc_df(df):
            df["text"] = Truthfullness.prompt_task_dict[prompt_task] + df["claim"]
            df['label'] = df['label'].astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train, valid, "factool", self.taskname, prompt_task)
        return train, valid
    
    def setup_truthfulqa(self, save=True, prompt_task=None):
        train = pd.read_csv(f"{data_dir}/base/truthfulqa_gen_train.csv")
        valid = pd.read_csv(f"{data_dir}/base/truthfulqa_gen_test.csv")
        def proc_df(df):
            df["text"] = Truthfullness.prompt_task_dict[prompt_task] + df["question"] + " " + df["claim"]
            df['label'] = df['label'].astype(int)
            return df[["idx", "text", "label"]]
        train = proc_df(train)
        valid = proc_df(valid)
        if save:
            save_dfs(train, valid, "truthfulqa_gen", self.taskname, prompt_task)
        return train, valid



def process_batch(batch_nos):
    if 0 in batch_nos:
        process_squad()
        process_healthver()
        process_selfaware()
        process_known_unkown()
        process_mmlu()
        process_real_toxicity_prompts()
        process_toxic_chat()
        process_qnota()
    if 1 in batch_nos:
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
    if 2 in batch_nos:
        process_cosmoqa()
        process_piqa()
        process_arc()
        process_medmcqa()
        process_commonsenseqa()
        process_openbookqa()
        process_qasc()
        process_hellaswag()
        process_bigbenchhard()
        process_truthfulqa()
    if 3 in batch_nos:
        process_climate_fever()
        process_felm()
        process_fever()
        process_averitec()
        process_factool()
        process_ragtruth()
        process_faithbench()




def process_all():
    process_batch([0, 1, 2, 3])
    return


def setup_batch(batch_nos):
    unanswerable = Unanswerable()
    toxicity = ToxicityAvoidance()
    jailbreak = Jailbreak()
    confidence = Confidence()
    news_topic = NewsTopic()
    sentiment = Sentiment()
    truthfulness = Truthfullness()
    if 0 in batch_nos:
        unanswerable.setuphealthver()
        unanswerable.setupqnota()
        unanswerable.setupselfaware()
        unanswerable.setupsquad()
        unanswerable.setupknown_unknown()
        toxicity.setup_real_toxicity_prompts()
        toxicity.setup_toxic_chat()
        jailbreak.setup_toxic_chat()
    if 1 in batch_nos:
        for key in news_topic.prompt_task_dict:
            news_topic.setupagnews(prompt_task=key)
            news_topic.setupbbcnews(prompt_task=key)
            news_topic.setupnytimes(prompt_task=key)
        for key in sentiment.prompt_task_dict:
            sentiment.setup_amazonreviews(prompt_task=key)
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
    if 2 in batch_nos:
        confidence.setup_mmlu()
        confidence.setup_cosmoqa()
        confidence.setup_piqa()
        confidence.setup_arc()
        confidence.setup_medmcqa()
        confidence.setup_commonsenseqa()
        confidence.setup_openbookqa()
        confidence.setup_qasc()
        confidence.setup_hellaswag()
        confidence.setup_bigbenchhard()
        confidence.setup_truthfulqa()
    if 3 in batch_nos:
        unanswerable.setupclimatefever()
        truthfulness.setup_felm()
        truthfulness.setup_healthver()
        truthfulness.setup_climate_fever()
        truthfulness.setup_averitec()
        truthfulness.setup_fever()
        truthfulness.setup_factool()
        truthfulness.setup_truthfulqa()
    return





def setup_all():
    setup_batch([0, 1, 2, 3])
    return



if __name__ == "__main__":
    process_batch([3])
    setup_batch([3])