tasks=("confidence")
datasets=("mmlu")
random_sample_train=1000
random_sample_test=1000

source proj_params.sh
for task in "${tasks[@]}"
do
    for dataset in "${datasets[@]}"
    do
        prediction_dir=$RESULTS_DIR/$task/predictions/$dataset/
        python run_iid.py --task $task --dataset $dataset --prediction_dir $prediction_dir --random_sample_train $random_sample_train --random_sample_test $random_sample_test
    done
done

prediction_dir=$RESULTS_DIR/$task/predictions/$dataset/
python run_iid.py --task $task --dataset $dataset --prediction_dir $prediction_dir --random_sample_train $random_sample_train --random_sample_test $random_sample_test