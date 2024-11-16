declare -A tasks_and_datasets=(
    ["confidence"]="mmlu cosmoqa piqa arc medmcqa commonsenseqa openbookqa qasc hellaswag bigbenchhard_mcq truthfulqa"
    ["nei"]="squad healthver"
    ["unanswerable"]="qnota selfaware known_unknown climatefever"
    ["sentiment"]="amazonreviews yelp twitterfinance twittermteb auditorsentiment fiqa indosentiment_eng newsmtc imdb financial_phrasebank dair_emotion sst5"
    ["truthfullness"]="felm healthver climatefever averitec fever factool truthfulqa_gen"
    ["newstopic"]="agnews bbcnews nytimes"
)
model_name="meta-llama/Llama-3.1-8B-Instruct"
random_seed=42
random_sample_train=2000
random_sample_test=10000000
model_kind="linear"

source proj_params.sh
model_save_name="${model_name#*/}"
for task in "${!tasks_and_datasets[@]}"
do
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        echo "XXXXXXXXXXXXXXX Running IID for $task $dataset XXXXXXXXXXXXXXXXXXX"
        prediction_dir=$RESULTS_DIR/$model_save_name/$task/predictions/$dataset/
        logfile=$LOG_DIR/probe_iid/$model_save_name/$task/$dataset/$model_kind/train_${random_sample_train}-test_${random_sample_test}-seed_${random_seed}.log
        mkdir -p $(dirname $logfile)
        python iid_modeling.py --task $task --dataset $dataset --model_save_name $model_save_name --prediction_dir $prediction_dir --random_sample_train $random_sample_train --random_sample_test $random_sample_test --random_seed $random_seed --model_kind $model_kind > $logfile
    done
done
