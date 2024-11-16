import os
from openai import OpenAI
import json
import time
import pandas as pd
import traceback
from vllm import LLM, SamplingParams

open_ai_save_path = os.environ["OPENAI_SAVE_PATH"] 

# TODO: 
# Implement the automatic splitting of batches for OpenAI
# Implement quantization for VLLM
# Give OpenAIGPT a single function that can be called and will either save a batch or load it and return the results if completed

class OpenAIGPT:
    def __init__(self, model="gpt-3.5-turbo", verbose=False, max_retries=3, default_max_tokens = 120):
        self.model = model
        self.verbose = verbose
        self.client = OpenAI()
        self.max_tokens = default_max_tokens
        self.max_retries = max_retries
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.prompt_cost = 0
        self.completion_cost = 0
        self.seconds_per_query = (60 / 20) + 0.01

    @staticmethod
    def openai_api_calculate_cost(message):
        model = message.model
        pricing = { # price per million tokens in USD
            'gpt-3.5-turbo-0125': {
                'prompt': 0.5,
                'completion': 1.5,
            },
            "gpt-4-turbo": {
                'prompt': 10,
                'completion': 30,
            },            
        }
        usage = message.usage
        model_pricing = pricing[model]
        million = 1_000_000
        prompt_cost = usage.prompt_tokens * (model_pricing['prompt'] /  million)
        completion_cost = usage.completion_tokens * (model_pricing['completion'] / million)
        return usage.prompt_tokens, usage.completion_tokens, prompt_cost, completion_cost

    def reset_usage_track(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.prompt_cost = 0
        self.completion_cost = 0

    def get_usage(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_cost": self.prompt_cost,
            "completion_cost": self.completion_cost,
            "total_cost": self.prompt_cost + self.completion_cost
        }

    def print_usage(self):
        usage = self.get_usage()
        print(f"Prompt tokens: {usage['prompt_tokens']}")
        print(f"Completion tokens: {usage['completion_tokens']}")
        print(f"Prompt cost: {usage['prompt_cost']}")
        print(f"Completion cost: {usage['completion_cost']}")
        print(f"Total cost: {usage['total_cost']}")

    def request_model(self, msgs):
        """
        msgs is a list with role and content: e.g. [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        """
        message = self.client.chat.completions.create(model=self.model, messages=msgs, stream=False, temperature=0)
        output = message.choices[0].message.content
        prompt_tokens, completion_tokens, prompt_cost, completion_cost = self.openai_api_calculate_cost(message)
        OpenAIGPT.prompt_tokens += prompt_tokens
        OpenAIGPT.completion_tokens += completion_tokens
        OpenAIGPT.prompt_cost += prompt_cost
        OpenAIGPT.completion_cost += completion_cost
        return output

    def execute(self, msgs, max_new_tokens=50):
        if self.verbose:
            for i, msg in enumerate(msgs):
                # print the role in green and text in blue
                # set print color to green
                color = "\033[92m" if i < len(msgs) - 1 else "\033[94m"
                print(color)
                print(f"{msg['role']}: {msg['content']}")
                # reset print color to default
                print("\033[0m")
        retries = 0
        self.max_tokens = max_new_tokens
        while retries < self.max_retries:
            try:
                output = self.request_model(msgs)
                if self.verbose:
                    print("\033[91m")
                    print(f"{output}")
                    print("\033[0m")
                return output
            except Exception as e:
                print(e)
                print(f"Sleeping for {self.seconds_per_query} seconds")
                time.sleep(self.seconds_per_query)
                retries += 1
            else:
                print(f"Shouldn't be getting here")
                retries = self.max_retries
        raise ValueError(f"Failed to get response after {self.max_retries} retries")
    
    def build_batch_file(self, nested_msg_list, batch_file_path, external_id=0):
        data = []
        columns = ["external_id", "internal_id", "method", "url", "body"]
        url = "/v1/chat/completions"
        method = "POST"
        for i, msgs in enumerate(nested_msg_list):
            custom_id = i
            body = {"model": self.model, "messages": msgs, "max_tokens": self.max_tokens}
            data.append([external_id, custom_id, method, url, body])
        df = pd.DataFrame(data, columns=columns)
        df.to_json(batch_file_path, lines=True, orient='records')
        return
    
    def send_batch(self, batch_file_path):
        batch_input_file = self.client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
            )
        batch_input_file_id = batch_input_file.id
        batch_data = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": "idk"
            }
        )
        batch_id = batch_data.id
        # write the batch_id and batch_input_file_id to a json file
        batch_info = {"batch_id": batch_id, "batch_input_file_id": batch_input_file_id}
        batch_info_path = batch_file_path.replace(".jsonl", "_info.json")
        with open(batch_info_path, "w") as f:
            json.dump(batch_info, f)
        return batch_info_path
    
    def do_batch(self, nested_msg_list, batch_file_path, external_id=0):
        self.build_batch_file(nested_msg_list, batch_file_path, external_id=external_id)
        batch_info_path = self.send_batch(batch_file_path)
        return batch_info_path
    
    def list_batches(self):
        batches = self.client.batches.list()
        return batches
    
    def get_batch(self, batch_file_path):
        batch_info_path = batch_file_path.replace(".jsonl", "_info.json")
        with open(batch_info_path, "r") as f:
            batch_info = json.load(f)
        batch_id = batch_info["batch_id"]
        batch = self.client.batches.retrieve(batch_id)
        return batch
    
    def get_batch_results(self, batch_file_path):
        batch = self.get_batch(batch_file_path)
        output_file_id = batch.output_file_id
        file_response = self.client.files.content(output_file_id)
        return file_response


    def read_results(self, filehash, split_into=5):
        batch_folder = f"{open_ai_save_path}/{filehash}/"
        parsed_file = f"{batch_folder}/parsed.csv"
        if os.path.exists(parsed_file):
            df = pd.read_csv(parsed_file)
            return df["text"].tolist()
        outputs = []
        for batch_file_id in range(split_into):
            batch_file_path = f"{batch_folder}/{batch_file_id}.jsonl"
            results = self.get_batch_results(batch_file_path)
            answer_df = pd.DataFrame(data=[json.loads(line) for line in results.text.strip().split('\n')])
            answer_df["text"] = answer_df["response"].apply(lambda x: x["body"]["choices"][0]["message"]["content"])
            answer_df["internal_id"] = answer_df["internal_id"].astype(int)
            answer_df["external_id"] = answer_df["external_id"].astype(int)
            assert answer_df["external_id"].nunique() == 1
            assert answer_df["internal_id"].nunique() == len(answer_df)
            assert answer_df["external_id"].unique()[0] == batch_file_id
            # sort by internal_id
            answer_df = answer_df.sort_values("internal_id").reset_index(drop=True)
            text_list = answer_df["text"].tolist()
            outputs.extend(text_list)
        df = pd.DataFrame(outputs, columns=["text"])
        df.to_csv(parsed_file, index=False)
        return outputs
        

    def __call__(self, nested_msg_list, filehash, split_into=5):
        batch_folder = f"{open_ai_save_path}/{filehash}/"
        if os.path.exists(batch_folder):
            try:
                outputs = self.read_results(filehash, split_into=split_into)
                return outputs
            except Exception as e:
                traceback.print_exc()
                print(f"Failed to read results from {batch_folder}")
        else:
            os.makedirs(batch_folder, exist_ok=True)
            for i in range(0, len(nested_msg_list), split_into):
                batch_file_path = f"{batch_folder}/{i}.jsonl"
                self.do_batch(nested_msg_list[i:i+split_into], batch_file_path, external_id=i)
            return None




import time

class VLLMModel:
    def __init__(self, model_name, max_new_tokens=10):
        self.model_name = model_name
        self.model = LLM(model_name)
        self.max_new_tokens = max_new_tokens
    
    def generate(self, prompts, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        sampling_params = SamplingParams(n=1, max_tokens=max_new_tokens)
        all_output = self.model.generate(prompts, sampling_params, use_tqdm=False)
        outputs = []
        for output in all_output:
            outputs.append(output.outputs[0].text)
        return outputs
    
    def __call__(self, prompts):
        return self.generate(prompts)


def get_prompt_engine(prompt_engine_name, max_new_tokens=10):
    if prompt_engine_name in ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]:
        return OpenAIGPT(model=prompt_engine_name, max_new_tokens=max_new_tokens)
    else:
        # assume its a valid VLLM model
        return VLLMModel(prompt_engine_name, max_new_tokens=max_new_tokens)
    raise ValueError(f"Prompt engine {prompt_engine_name} not found.")