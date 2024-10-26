python save_inference.py --model_name meta-llama/Llama-3.1-8B-Instruct --data_path data/unanswerable/healthver_train.csv --output_csv_path results/unanswerable/healthver_train_inference.csv --output_hidden_dir results/unanswerable/train/healthver --max_new_tokens 1
python save_inference.py --model_name meta-llama/Llama-3.1-8B-Instruct --data_path data/unanswerable/healthver_test.csv --output_csv_path results/unanswerable/healthver_test_inference.csv --output_hidden_dir results/unanswerable/test/healthver --max_new_tokens 1

python save_inference.py --model_name meta-llama/Llama-3.1-8B-Instruct --data_path data/unanswerable/selfaware_test.csv --output_csv_path results/unanswerable/selfaware_test_inference.csv --output_hidden_dir results/unanswerable/test/selfaware --max_new_tokens 1
python save_inference.py --model_name meta-llama/Llama-3.1-8B-Instruct --data_path data/unanswerable/selfaware_train.csv --output_csv_path results/unanswerable/selfaware_train_inference.csv --output_hidden_dir results/unanswerable/train/selfaware --max_new_tokens 1

python save_inference.py --model_name meta-llama/Llama-3.1-8B-Instruct --data_path data/unanswerable/squad_train.csv --output_csv_path results/unanswerable/squad_train_inference.csv --output_hidden_dir results/unanswerable/train/squad --max_new_tokens 1
python save_inference.py --model_name meta-llama/Llama-3.1-8B-Instruct --data_path data/unanswerable/squad_test.csv --output_csv_path results/unanswerable/squad_test_inference.csv --output_hidden_dir results/unanswerable/test/squad --max_new_tokens 1

python save_inference.py --model_name meta-llama/Llama-3.1-8B-Instruct --data_path data/unanswerable/qnota_train.csv --output_csv_path results/unanswerable/qnota_train_inference.csv --output_hidden_dir results/unanswerable/train/qnota --max_new_tokens 1
python save_inference.py --model_name meta-llama/Llama-3.1-8B-Instruct --data_path data/unanswerable/qnota_test.csv --output_csv_path results/unanswerable/qnota_test_inference.csv --output_hidden_dir results/unanswerable/test/qnota --max_new_tokens 1 