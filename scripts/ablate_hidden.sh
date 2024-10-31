declare -A tasks_and_datasets=(
    ["confidence"]="mmlu healthver"
) # add a space between dataset names for each task
random_sample_train=1000
random_sample_test=1000

source proj_params.sh

for task in "${!tasks_and_datasets[@]}"
do
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        echo "XXXXXXXXXXXXXXXXXXXX Running IID for $task $dataset XXXXXXXXXXXXXXXXXXXXXX"
        prediction_dir=$RESULTS_DIR/$task/predictions/$dataset/
        common="--task $task --dataset $dataset --prediction_dir $prediction_dir --random_sample_train $random_sample_train --random_sample_test $random_sample_test"
        # Using only layer_2
        python iid_modeling.py --exclude_layers layer_15 --exclude_layers layer_30  $common
        # Using only layer_15
        python iid_modeling.py --exclude_layers layer_2 --exclude_layers layer_30  $common
        # Using only layer_30
        python iid_modeling.py --exclude_layers layer_2 --exclude_layers layer_15  $common
        # Using layer_2 and layer_15
        python iid_modeling.py --exclude_layers layer_30  $common
        # Using layer_2 and layer_30
        python iid_modeling.py --exclude_layers layer_15  $common
        # Using layer_15 and layer_30
        python iid_modeling.py --exclude_layers layer_2  $common
        # Using all layers
        python iid_modeling.py $common
        # Using only mlp
        python iid_modeling.py --exclude_hidden attention --exclude_hidden projection $common
        # Using only attention
        python iid_modeling.py --exclude_hidden mlp --exclude_hidden projection $common
        # Using only projection
        python iid_modeling.py --exclude_hidden mlp --exclude_hidden attention $common
        # Using mlp and attention
        python iid_modeling.py --exclude_hidden projection $common
        # Using mlp and projection
        python iid_modeling.py --exclude_hidden attention $common
        # Using attention and projection
        python iid_modeling.py --exclude_hidden mlp $common
        # Using task offset
        python iid_modeling.py --use_task_offset True $common
    done
done
