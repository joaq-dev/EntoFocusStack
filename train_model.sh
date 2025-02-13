CUDA_VISIBLE_DEVICES=0,4 torchrun --nproc_per_node=2  main.py --mode train --data_dir "data/focus_stack_dataset/dataset"  --epochs 1000
