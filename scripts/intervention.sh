declare -A tasks_and_datasets=(
    ["confidence"]="mmlu cosmoqa piqa arc medmcqa commonsenseqa openbookqa qasc hellaswag bigbenchhard_mcq truthfulqa"
    ["sentiment"]="amazonreviews yelp twitterfinance twittermteb auditorsentiment fiqa indosentiment_eng newsmtc imdb financial_phrasebank dair_emotion sst5"
    ["truthfullness"]="felm healthver climatefever averitec fever factool truthfulqa_gen"
    ["newstopic"]="agnews bbcnews nytimes"
)

model_kind="linear"
intervention_strength=0.1
intervention_layer=15
intervention_location="mlp"
splits=("train" "test")
model_name="meta-llama/Llama-3.1-8B-Instruct"
source_run_name="iid"
source_random_seed=42
source_random_sample_train=2000
source_model_kind="linear"



declare -A max_new_token_dict=( ["confidence"]=5 ["unanswerable"]=10 ["toxicity_avoidance"]=20 ["sentiment"]=10 ["truthfullness"]=10 ["nei"]=10 ["newstopic"]=10)
source proj_params.sh
model_save_name="${model_name#*/}"
source_weight_path_start=$RESULTS_DIR/$model_save_name/predictions/$source_run_name/

for task in "${!tasks_and_datasets[@]}"
do
    max_new_tokens=${max_new_token_dict[$task]}
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        intervention_vector_path=$source_weight_path_start/$task/$dataset/$source_model_kind/train_${source_random_sample_train}-seed_${source_random_seed}/model.npy
        for split in "${splits[@]}"
        do
            echo "XXXXXXXXXXXXXXXX $task $dataset $split XXXXXXXXXXXXXXXX"
            data_path="$DATA_DIR/$task/${dataset}_$split.csv"
            output_csv_path="$RESULTS_DIR/$model_save_name/$task/${dataset}_${split}_intervention.csv"
            python perform_intervention.py --model_name $model_name --data_path $data_path --output_csv_path $output_csv_path --intervention_vector_path $intervention_vector_path --intervention_layer $intervention_layer --intervention_strength $intervention_strength --intervention_location $intervention_location --max_new_tokens $max_new_tokens
        done
    done
done