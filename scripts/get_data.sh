source proj_params.sh
cd $PROJECT_STORE_DIR
mkdir data
cd data
mkdir raw
cd raw
mkdir known_unknown
cd known_unknown
gdown https://drive.google.com/file/d/1BJv5Uqy1QnX_tcZ4jkn5QAtI2XIhec_z/view?usp=sharing --fuzzy
gdown https://drive.google.com/file/d/1Lmgqn_YZBGKeSDloBzxhH0NnpGk2WRyH/view?usp=sharing --fuzzy
cd ..
mkdir healthver
cd healthver
wget https://raw.githubusercontent.com/sarrouti/HealthVer/refs/heads/master/data/healthver_train.csv
wget https://raw.githubusercontent.com/sarrouti/HealthVer/refs/heads/master/data/healthver_dev.csv
wget https://raw.githubusercontent.com/sarrouti/HealthVer/refs/heads/master/data/healthver_test.csv
cd ..
mkdir qnota
cd qnota
wget https://raw.githubusercontent.com/Nisarg-P-Patel/QNotA_dataset/refs/heads/main/Dataset/incomplete_questions.json
wget https://raw.githubusercontent.com/Nisarg-P-Patel/QNotA_dataset/refs/heads/main/Dataset/futuristic_questions.json
wget https://raw.githubusercontent.com/Nisarg-P-Patel/QNotA_dataset/refs/heads/main/Dataset/ambiguous_questions.json
wget https://raw.githubusercontent.com/Nisarg-P-Patel/QNotA_dataset/refs/heads/main/Dataset/unmeasurable_questions.json
wget https://raw.githubusercontent.com/Nisarg-P-Patel/QNotA_dataset/refs/heads/main/Dataset/incorrect_questions.json

mkdir nytimes
cd nytimes
wget https://raw.githubusercontent.com/billywzh717/N24News/refs/heads/main/nytimes_dataset.json