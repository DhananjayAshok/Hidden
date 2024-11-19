sentstring="amazonreviews yelp twitterfinance twittermteb auditorsentiment fiqa indosentiment_eng newsmtc imdb financial_phrasebank dair_emotion sst5"
newstring="agnews bbcnews nytimes"
truthstring="felm healthver climatefever averitec fever factool truthfulqa_gen"
declare -A tasks_and_datasets=(
    ["newstopic_summ"]=$newstring
    ["newstopic_question"]=$newstring
    ["newstopic_answer"]=$newstring
    ["newstopic_topic"]=$newstring
    ["sentiment_speaker"]=$sentstring
    ["sentiment_question"]=$sentstring
    ["sentiment_answer"]=$sentstring
    ["sentiment_sentiment"]=$sentstring
    ["truthfullness_speaker"]=$truthstring
    ["truthfullness_truth"]=$truthstring
)

model_name="meta-llama/Llama-3.1-8B-Instruct"
random_seed=42
random_sample_train=2000
random_sample_test=10000000
model_kind="linear"
only_mlp=True
only_attention=False
only_layer=15

source proj_params.sh
model_save_name="${model_name#*/}"
for task in "${!tasks_and_datasets[@]}"
do
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        echo "XXXXXXXXXXXXXXX Running IID for $task $dataset XXXXXXXXXXXXXXXXXXX"
        middle=/$task/$dataset/$model_kind/
        prediction_dir=$RESULTS_DIR/$model_save_name/predictions/iid/$middle/train_${random_sample_train}-seed_${random_seed}/
        logfile=$LOG_DIR/probe_iid/$model_save_name/$middle/train_${random_sample_train}-test_${random_sample_test}-seed_${random_seed}.log
        mkdir -p $(dirname $logfile)
        python iid_modeling.py --task $task --dataset $dataset --model_save_name $model_save_name --prediction_dir $prediction_dir --random_sample_train $random_sample_train --random_sample_test $random_sample_test --random_seed $random_seed --model_kind $model_kind --only_attention $only_attention --only_mlp $only_mlp --only_layer $only_layer > $logfile
    done
done
