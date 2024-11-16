from transformers import pipeline
import click
import pandas as pd
from tqdm import tqdm
import warnings
import os
from prompt_engines import get_prompt_engine, OpenAIGPT, VLLMModel
import hashlib
import numpy as np


def generate_hash(data, max_length=10):
    """Generates a SHA-256 hash of the input data."""

    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()

    # Update the hash object with the data
    if isinstance(data, str):
        data = data.encode('utf-8')
    hash_object.update(data)

    # Return the hexadecimal representation of the hash
    return hash_object.hexdigest()[:max_length]

@click.command()
@click.option("--file", type=str, required=True)
@click.option("--metric_name", type=str, required=True)
@click.option("--prompt_column", type=str, default="text")
@click.option("--generation_column", type=str, default="output")
@click.option('--reference_column', type=str, default=None)
@click.option("--output_column", type=str, default="label")
@click.option("--use_prompt", type=bool, default=False)
@click.option("--overwrite", type=bool, default=False)
@click.option('--binarize_threshold', type=float, default=None)
def main(file, metric_name, prompt_column, generation_column, reference_column, output_column, use_prompt, overwrite, binarize_threshold):
    if reference_column == "none":
        reference_column = None
    try:
        df = pd.read_csv(file)
    except:
        df = pd.read_csv(file, lineterminator="\n")
    if generation_column not in df.columns:
        raise ValueError(f"Input column {generation_column} not found in dataframe with columns {df.columns}")
    if reference_column is not None and reference_column not in df.columns:
        raise ValueError(f"Reference column {reference_column} not found in dataframe with columns {df.columns}")
    if use_prompt and prompt_column not in df.columns:
        raise ValueError(f"Prompt column {prompt_column} not found in dataframe with columns {df.columns}")
    if output_column in df.columns and not overwrite:
        raise ValueError(f"Output column {output_column} already found in dataframe with columns {df.columns}. Set overwrite to True to overwrite")
    to_score = df[generation_column]
    labels = df[reference_column] if reference_column is not None else [None for _ in range(len(df))]
    if use_prompt:
        to_score = df[prompt_column] + " " + to_score
    metric = get_metric_class(metric_name)
    indices = []
    texts = []
    selected_labels = []
    for i in range(len(df)):
        if not isinstance(to_score[i], str):
            continue
        indices.append(i)
        texts.append(to_score[i])
        selected_labels.append(labels[i])
    filehash = generate_hash(metric_name+file)
    outputs = metric(texts, selected_labels, filehash)
    for i, idx in enumerate(indices):
        df.loc[idx, output_column] = outputs[i]
    if binarize_threshold is not None and df[output_column].dtype == "float":
        df[f"{output_column}_cts"] = df[output_column]
        df[output_column] = df[output_column] > binarize_threshold
    df.to_csv(file, index=False)


def get_metric_class(metric_name):
    """
    Returns a class that can be called with input text and outputs a single float value
    """
    if "-" in metric_name:
        prompt_engine = "-".join(metric_name.split("-")[1:])
        metric_name = metric_name.split("-")[0]
    if metric_name == "toxdectroberta":
        return ToxDectRoberta()
    if metric_name == "match":
        return StringMatch()
    if metric_name == "matchcot":
        return StringMatch(cot=True)
    if metric_name == "mcqmatch":
        return MCQStringMatch()
    if metric_name == "mcqmatchcot":
        return MCQStringMatch(cot=True)
    if metric_name == "unanswerable":
        return UnanswerablePseudoLabel(prompt_engine)
    if metric_name == "fewshot_tf":
        return FewShotTFPsuedoLabel(prompt_engine)

    raise ValueError(f"Metric {metric_name} not found")

class ToxDectRoberta:
    def __init__(self, batch_size=5):
        self.batch_size = batch_size
        self.classifier = pipeline("text-classification", model='Xuhui/ToxDect-roberta-large', return_all_scores=True, device_map="auto")

    def __call__(self, texts, labels, filehash=None):
        all_scores = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i+self.batch_size]
            batch_scores = self.classifier(batch_texts)
            for output in batch_scores:
                scores = [0, 0]
                for scoredict in output:
                    label = scoredict["label"]
                    score = scoredict["score"]
                    if label == "LABEL_0":
                        scores[0] = score
                    elif label == "LABEL_1":
                        scores[1] = score
                all_scores.append(scores[1])
        return all_scores


class HallucinationDetector:
    def __init__(self, batch_size=5):
        self.batch_size = batch_size
        self.classifier = pipeline("text-classification", model="vectara/hallucination_evaluation_model", trust_remote_code=True, device_map="auto")


    @staticmethod
    def get_context(text):
        return None
    
    @staticmethod
    def get_generation(text):
        return None

    def __call__(self, texts, labels, filehash=None):        
        prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: [TEXT1]\n\nHypothesis: [TEXT2]"
        all_scores = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = [x.replace("[TEXT1]", self.get_context(x)).replace("[TEXT2]", self.get_generation(x)) for x in texts[i:i+self.batch_size]]
            batch_scores = self.classifier(batch_texts, top_k=None)
            simple_scores = [score_dict['score'] for score_for_both_labels in batch_scores for score_dict in score_for_both_labels if score_dict['label'] == 'hallucinated']
            for output in batch_scores:
                scores = [0, 0]
                for scoredict in output:
                    label = scoredict["label"]
                    score = scoredict["score"]
                    if label == "LABEL_0":
                        scores[0] = score
                    elif label == "LABEL_1":
                        scores[1] = score
                all_scores.append(scores[1])
        return all_scores


class StringMatch:
    def __init__(self, cot=False):
        self.cot = cot

    def __call__(self, texts, labels, filehash=None):
        scores = []
        check_fn = self.cot_check if self.cot else self.normal_check
        for text, label in tqdm(zip(texts, labels), total=len(texts)):
            scores.append(check_fn(text, label))
        return scores
    
    @staticmethod
    def cot_check(prediction, label):
        if prediction is None or label is None:
            return None
        pred_labels = prediction.split("||")
        if len(pred_labels) == 1:
            warnings.warn(f"Supposed to be COT but prediction {prediction} does not have ||")
            pred_labels = pred_labels[0]
        elif len(pred_labels) == 2:
            pred_labels = pred_labels[1]
        else:
            warnings.warn(f"Supposed to be COT but prediction {prediction} has more than 1 ||")
            pred_labels = pred_labels[1]
        return StringMatch.normal_check(pred_labels, label)

    @staticmethod
    def normal_check(prediction, label):
        if prediction is None or label is None:
            return None
        return prediction.lower().strip() == label.lower().strip()

class MCQStringMatch(StringMatch):
    def __init__(self, cot=False):
        super().__init__(cot=cot)
        self.label_map_dict = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5", "F": "6", "G": "7", "H": "8", "I": "9", "J": "10"}

    def __call__(self, texts, labels, filehash=None):
        scores = []
        check_fn = self.cot_check if self.cot else self.normal_check
        for text, label in tqdm(zip(texts, labels), total=len(texts)):
            op1 = check_fn(text, label)
            if label in self.label_map_dict:
                op2 = check_fn(text, self.label_map_dict[label])
            else:
                op2 = False
            scores.append(op1 or op2)
        return scores



class PromptHolder:
    def __init__(self, system_prompt, instance_queries, instance_answers):
        self.system_prompt = system_prompt
        self.instance_queries = instance_queries
        self.instance_answers = instance_answers

    def __call__(self, new_query, k=3, openai=False):
        n_options = len(self.instance_queries)
        random_indices = np.random.choice(range(n_options), k, replace=False)
        assert n_options > k
        if openai:
            msgs = [] # append a system role system prompt
            msgs.append({"role": "system", "content": self.system_prompt})
            for idx in random_indices:
                msgs.append({"role": "user", "content": self.instance_queries[idx]})
                msgs.append({"role": "assistant", "content": self.instance_answers[idx]})
            return msgs
        else:
            prompt = self.system_prompt+"\n"
            for idx in random_indices:
                prompt += "Q: " + self.instance_queries[idx] + "\nAnswer: "
                prompt += self.instance_answers[idx] + "\n"
            prompt += "Q: "+ new_query+"\nAnswer: "
            return prompt


class UnanswerablePromptHolder(PromptHolder):
    def __init__(self):
        system_prompt = "Answer with only YES or NO. Is the following question unanswerable?"
        instance_queries = ["What is the capital of France?", "What is the size of the largest planet in the Milky Way?", "What was the name of the first human?", "Who is the current president of the United States?"]
        instance_answers = ["NO", "YES", "YES", "NO"]
        super().__init__(system_prompt, instance_queries, instance_answers)

class PseudoLabel:
    def __init__(self, prompt_engine_name, batch_size):
        self.prompt_engine = get_prompt_engine(prompt_engine_name)
        self.batch_size = batch_size

    def __call__(self, texts, labels, filehash=None):
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = len(texts)
        scores = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            if filehash is None:
                actual_hash = None
            else:
                actual_hash = filehash+f"dfbatch_{i}"
            batch_prompts = [self.get_prompt(text, label) for text, label in zip(batch_texts, batch_labels)]
            if isinstance(self.prompt_engine, OpenAIGPT):
                batch_outputs = self.prompt_engine(batch_prompts, actual_hash)
            else:
                batch_outputs = self.prompt_engine(batch_prompts)
            batch_scores = [self.parse_fn(output) for output in batch_outputs]
            scores.extend(batch_scores)
        return scores
   
    @staticmethod
    def parse_yes_no(output):
        if not isinstance(output, str):
            return None
        output = output.lower().strip()
        for positive in ["yes", "true", "1"]:
            if positive in output:
                return 1
        for negative in ["no", "false", "0"]:
            if negative in output:
                return 0
        return None
    
    @staticmethod
    def parse_score(output, min_score=0, max_score=10, norm=True):
        divisor = 1 if not norm else (max_score - min_score)
        assert divisor > 0
        try:
            value = int(output)
            if value < min_score:
                return min_score / divisor
            if value > max_score:
                return max_score / divisor
            return value            
        except:
            for i in range(max_score, min_score-1, -1):
                if str(i) in output:
                    return i / divisor
            return None


    def get_prompt(self, text, label):
        raise NotImplementedError
    

class UnanswerablePseudoLabel(PseudoLabel):
    def __init__(self, prompt_engine_name, batch_size=None):
        super().__init__(prompt_engine_name, batch_size)
        self.prompt_holder = UnanswerablePromptHolder()
        self.parse_fn = self.parse_yes_no


    def get_prompt(self, text, label):
        new_query = text
        is_openai = isinstance(self.prompt_engine, OpenAIGPT)
        return self.prompt_holder(new_query, openai=is_openai)


class FewShotTFPsuedoLabel(PseudoLabel):
    def __init__(self, prompt_engine_name, batch_size=None):
        super().__init__(prompt_engine_name, batch_size)
        self.parse_fn = self.parse_yes_no

    def get_prompt(self, text, label):
        new_query = text
        is_openai = isinstance(self.prompt_engine, OpenAIGPT)
        assert not is_openai, f"Haven't implemented this yet"
        return new_query


if __name__ == "__main__":
    main()


