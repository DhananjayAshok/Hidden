run_name="IDK"
declare -A tasks_and_datasets=(
    ["confidence"]="mmlu cosmoqa piqa arc medmcqa commonsenseqa openbookqa qasc hellaswag bigbenchhard_mcq truthfulqa"
    ["nei"]="squad healthver"
    ["unanswerable"]="qnota selfaware known_unknown climatefever"
    ["sentiment"]="amazonreviews yelp twitterfinance twittermteb auditorsentiment fiqa indosentiment_eng newsmtc imdb financial_phrasebank dair_emotion sst5"
    ["truthfullness"]="felm healthver climatefever averitec fever factool truthfulqa_gen"
    ["newstopic"]="agnews bbcnews nytimes"
)
strict_w_dataset=False
strict_w_task=False
mix_iid_n=0
random_sample_train_per=200
model_name="meta-llama/Llama-3.1-8B-Instruct"
model_kind="linear"
random_sample_test_per=200000
random_seed=42


taskdata=""

for task in "${!tasks_and_datasets[@]}"
do
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        taskdata="$taskdata --task_datasets $task $dataset"
    done
done


source proj_params.sh
model_save_name="${model_name#*/}"
logfile=$LOG_DIR/probe_ood/$model_save_name/$run_name/$model_kind/train_${random_sample_train_per}-seed_${random_seed}.log
echo "XXXXXXXXXXXXXXX Running OOD for $task XXXXXXXXXXXXXXXXXXX"
python ood_modeling.py $taskdata --model_save_name $model_save_name --random_sample_train_per $random_sample_train_per --random_sample_test_per $random_sample_test_per --random_seed $random_seed --strict_w_dataset $strict_w_dataset --strict_w_task $strict_w_task --mix_iid_n $mix_iid_n --model_kind $model_kind > $logfile
