import pandas as pd
import os

results_dir = os.getenv("RESULTS_DIR")
data_dir = os.getenv("DATA_DIR")


def mmlu_output_map(x):
    if "A" in x:
        return "A"
    elif "B" in x:
        return "B"
    elif "C" in x:
        return "C"
    elif "D" in x:
        return "D"
    else:
        return None
    
def process_mmlu_result(mmlu_path):
    df = pd.read_csv(mmlu_path)
    df["output_parsed"] = df["output"].apply(mmlu_output_map)
    df["correct"] = df["output_parsed"] == df["label"]
    df.to_csv(mmlu_path, index=False)

def process_mmlu():
    mmlu_path = f"{results_dir}/confidence/mmlu_train_inference.csv"
    process_mmlu_result(mmlu_path)
    mmlu_path = f"{results_dir}/confidence/mmlu_test_inference.csv"
    process_mmlu_result(mmlu_path)
    return

if __name__ == "__main__":
    process_mmlu()