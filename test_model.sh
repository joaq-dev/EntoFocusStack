#!/bin/bash

# First evaluation
python main.py --mode eval --data_dir "data/focus_stack_dataset/dataset" --train_dir "//timestamped train folder" --dataset_lens firstgroup

# Second evaluation
python main.py --mode eval --data_dir "data/focus_stack_dataset/dataset" --train_dir "//timestamped train folder" --dataset_lens secondgroup

