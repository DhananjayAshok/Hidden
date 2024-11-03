declare -A tasks_and_datasets=(
    ["confidence"]="mmlu healthver"
)
random_seed=42
random_sample_train=1000
random_sample_test=1000

source proj_params.sh

for task in "${!tasks_and_datasets[@]}"
do
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        echo "XXXXXXXXXXXXXXX Running IID for $task $dataset XXXXXXXXXXXXXXXXXXX"
        prediction_dir=$RESULTS_DIR/$task/predictions/$dataset/
        python iid_modeling.py --task $task --dataset $dataset --prediction_dir $prediction_dir --random_sample_train $random_sample_train --random_sample_test $random_sample_test --random_seed $random_seed
    done
done
