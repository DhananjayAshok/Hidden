cd data/raw
mkdir known_unknown
cd known_unknown
gdown https://drive.google.com/file/d/1BJv5Uqy1QnX_tcZ4jkn5QAtI2XIhec_z/view?usp=sharing --fuzzy
gdown https://drive.google.com/file/d/1Lmgqn_YZBGKeSDloBzxhH0NnpGk2WRyH/view?usp=sharing --fuzzy
cd ..
mkdir healthver
wget https://raw.githubusercontent.com/sarrouti/HealthVer/refs/heads/master/data/healthver_train.csv
wget https://raw.githubusercontent.com/sarrouti/HealthVer/refs/heads/master/data/healthver_dev.csv
wget https://raw.githubusercontent.com/sarrouti/HealthVer/refs/heads/master/data/healthver_test.csv