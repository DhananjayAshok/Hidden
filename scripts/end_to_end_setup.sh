git submodule init
git submodule update
conda create -n hidden python=3.10
conda activate hidden
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
cd transformers
pip install -e .
cd ..
pip install click, gdown, datasets, accelerate
bash scripts/get_data.sh
python data.py 