run_name_base="ood-default"
strict_w_dataset=False
strict_w_task=False
mix_iid_n=0
ood_config_files=("confidence" "nei" "unanswerable" "sentiment" "truthfullness" "newstopic")
random_sample_train_per=200
model_name="meta-llama/Llama-3.1-8B-Instruct"
model_kind="linear"
source proj_params.sh
random_sample_test_per=200000
random_seed=42
model_save_name="${model_name#*/}"


for ood_config_file in "${ood_config_files[@]}"
do
    source scripts/$run_name_base-config/${ood_config_file}.sh # this defines the key and tasks_and_datasets variables
    run_name="${run_name_base}_${key}"
    taskdata=""
    for task in "${!tasks_and_datasets[@]}"
    do
        datasets=${tasks_and_datasets[$task]}
        for dataset in $datasets
        do
            taskdata="$taskdata --task_datasets $task $dataset"
        done
    done
    logfile=$LOG_DIR/probe_ood/$model_save_name/$run_name/$model_kind/train_${random_sample_train_per}-seed_${random_seed}.log
    mkdir -p $(dirname $logfile)
    python ood_modeling.py --run_name_base $run_name_base $taskdata --model_save_name $model_save_name --random_sample_train_per $random_sample_train_per --random_sample_test_per $random_sample_test_per --random_seed $random_seed --strict_w_dataset $strict_w_dataset --strict_w_task $strict_w_task --mix_iid_n $mix_iid_n --model_kind $model_kind > $logfile
done