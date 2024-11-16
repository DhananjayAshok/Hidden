declare -A tasks_and_datasets=(
    ["confidence"]="mmlu cosmoqa piqa arc medmcqa commonsenseqa openbookqa qasc hellaswag bigbenchhard_mcq truthfulqa"
    ["nei"]="squad healthver"
    ["unanswerable"]="qnota selfaware known_unknown"
    ["sentiment"]="amazonreviews yelp twitterfinance twittermteb auditorsentiment fiqa indosentiment_eng newsmtc imdb financial_phrasebank dair_emotion sst5"
    ["truthfullness"]="felm healthver climatefever averitec fever factool truthfulqa_gen"
    ["newstopic"]="agnews bbcnews nytimes"
)