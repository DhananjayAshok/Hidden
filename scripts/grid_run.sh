declare -A tasks_and_datasets=(
    ["confidence"]="mmlu"
    ["unanswerable"]="squad healthver qnota selfaware known_unknown"
)
model_name="meta-llama/Llama-3.1-8B-Instruct"


source proj_params.sh
model_save_name="${model_name#*/}"
for task in "${!tasks_and_datasets[@]}"
do
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        python grid.py --task $task --dataset $dataset --model_save_name $model_save_name --n_samples 1000
    done
done
