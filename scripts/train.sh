file_path=$(readlink -f "$(dirname "$0")")
cd $file_path/..
python train.py --dataset TianChi --learning_rate 0.003 --gpu_id 0 --n_valid 100 --patient 2 --n_walks 10 --walk_length 3 \
  --batch_size 1024 --max_iter 50 --valid_interval 50 --time_agg 345600