tasks=("unanswerable")
random_seed=42
model_name="meta-llama/Llama-3.1-8B-Instruct"
random_sample_train_per=200
random_sample_test_per=2000

source proj_params.sh
model_save_name="${model_name#*/}"
for task in "${tasks[@]}"
do
    prediction_dir=$RESULTS_DIR/$model_save_name/$task/predictions/ood/
    echo "XXXXXXXXXXXXXXX Running OOD for $task XXXXXXXXXXXXXXXXXXX"
    python ood_modeling.py --task $task --model_save_name $model_save_name --prediction_dir $prediction_dir --random_sample_train_per $random_sample_train_per --random_sample_test_per $random_sample_test_per --random_seed $random_seed
done