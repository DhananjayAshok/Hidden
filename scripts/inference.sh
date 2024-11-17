declare -A tasks_and_datasets=(
    ["confidence"]="mmlu cosmoqa piqa arc medmcqa commonsenseqa openbookqa qasc hellaswag bigbenchhard_mcq truthfulqa"
    ["nei"]="squad healthver"
    ["unanswerable"]="qnota selfaware known_unknown climatefever"
    ["sentiment"]="amazonreviews yelp twitterfinance twittermteb auditorsentiment fiqa indosentiment_eng newsmtc imdb financial_phrasebank dair_emotion sst5"
    ["truthfullness"]="felm healthver climatefever averitec fever factool truthfulqa_gen"
    ["newstopic"]="agnews bbcnews nytimes"
)
splits=("train" "test")
model_name="meta-llama/Llama-3.1-8B-Instruct"



declare -A max_new_token_dict=( ["confidence"]=5 ["unanswerable"]=1 ["toxicity_avoidance"]=20 ["sentiment"]=1 ["truthfullness"]=1 ["nei"]=1 ["newstopic"]=1)
source proj_params.sh
model_save_name="${model_name#*/}"


for task in "${!tasks_and_datasets[@]}"
do
    max_new_tokens=${max_new_token_dict[$task]}
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        for split in "${splits[@]}"
        do
            echo "XXXXXXXXXXXXXXXX $task $dataset $split XXXXXXXXXXXXXXXX"
            data_path="$DATA_DIR/$task/${dataset}_$split.csv"
            output_csv_path="$RESULTS_DIR/$model_save_name/$task/${dataset}_${split}_inference.csv"
            output_hidden_dir="$RESULTS_DIR/$model_save_name/$task/$split/${dataset}"
            python save_inference.py --model_name $model_name --data_path $data_path --output_csv_path $output_csv_path --output_hidden_dir $output_hidden_dir --max_new_tokens $max_new_tokens
        done
    done
done