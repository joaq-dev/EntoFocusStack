#!/bin/bash

# First evaluation
CUDA_VISIBLE_DEVICES=0,4 python main.py --mode eval --data_dir data --train_dir "//timestamped train folder" --dataset_lens firstgroup

# Second evaluation
CUDA_VISIBLE_DEVICES=0,4 python main.py --mode eval --data_dir data --train_dir "//timestamped train folder" --dataset_lens secondgroup

