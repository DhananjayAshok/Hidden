declare -A tasks_and_datasets=(
    ["confidence"]="mmlu cosmoqa piqa arc medmcqa commonsenseqa openbookqa qasc hellaswag bigbenchhard_mcq truthfulqa"
    ["nei"]="squad healthver"
    ["unanswerable"]="qnota selfaware known_unknown"
    ["sentiment"]="amazonreviews yelp twitterfinance twittermteb auditorsentiment fiqa indosentiment_eng newsmtc imdb financial_phrasebank dair_emotion sst5"
    ["truthfullness"]="felm healthver climatefever averitec fever factool truthfulqa_gen"
)
model_name="meta-llama/Llama-3.1-8B-Instruct"
reference_column="none"
output_column="fewshot_pred"
metric_name="fewshot_tf-$model_name"
use_prompt=False

model_save_name="${model_name#*/}"
source proj_params.sh
for task in "${!tasks_and_datasets[@]}"
do
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        echo "XXXXXXXXXXXXXXX Running FewShot-TF Score Gen for $task $dataset XXXXXXXXXXXXXXXXXXX"
        filename="$DATA_DIR/fewshot_eval/$model_save_name/$task/${dataset}_test.csv"
        python score_gen.py --file $filename --reference_column $reference_column --output_column $output_column --use_prompt $use_prompt --metric_name $metric_name
    done
done