datasets=("real_toxicity_prompt" "toxic_chat")
task="toxicity_avoidance"
metric_name="toxdectroberta"
model_name="meta-llama/Llama-3.1-8B-Instruct"
#model_name="google/gemma-2-9b-it"
#model_name="mistralai/Ministral-8B-Instruct-2410"
reference_column="none"
output_column="label"
use_prompt=False
splits=("train" "test")
source proj_params.sh

model_name="${model_name#*/}"

for dataset in "${datasets[@]}"
    for split in "${splits[@]}"
    do

        filename="$RESULTS_DIR/$model_name/$task/${dataset}_${split}_inference.csv"
        python score_gen.py --file $filename --reference_column $reference_column --output_column $output_column --use_prompt $use_prompt --metric_name $metric_name
    done 
done