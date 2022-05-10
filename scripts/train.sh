file_path=$(readlink -f "$(dirname "$0")")
cd $file_path/..

# Running on the UCI dataset
python train.py --dataset uci --walk_length 1

# Running on the Amazon dataset
python train.py --dataset amazon --max_iter 20 --walk_length 5

# Running on the Last.fm dataset
python train.py --dataset Lastfm --max_iter 10 --n_negative 1 --time_agg 6192000000 --walk_length 4 --n_walks 12 --valid_interval 0 --weight_decay 0.00008 --learning_rate 0.001

# Running on the MovieLens dataset
python train.py --dataset MovieLens --max_iter 4 --n_negative 5 --walk_length 1 --n_walks 1 --valid_interval 0 --weight_decay 0.00008 --learning_rate 0.001

# Running on the Taobao dataset
python train.py --dataset Taobao
