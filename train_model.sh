CUDA_VISIBLE_DEVICES=0,4 torchrun --nproc_per_node=2  main.py --mode train --data_dir data  --epochs 1000 --n_colors 3
