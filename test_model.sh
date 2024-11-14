#!/bin/bash

# First evaluation
CUDA_VISIBLE_DEVICES=0,6 python main.py --mode eval --local --data_dir "data/focus_stack_dataset/dataset" --train_dir "//timestamped train folder" --dataset_lens firstgroup

# Second evaluation
CUDA_VISIBLE_DEVICES=0,6 python main.py --mode eval --local --data_dir "data/focus_stack_dataset/dataset" --train_dir "//timestamped train folder" --dataset_lens secondgroup

