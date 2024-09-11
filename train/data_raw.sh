

# e.g., real-world setting for GATEAU-LLaMA-7B-5K
python pre_tokenize.py --input_long_dir ./data/gateau_long.jsonl  --input_share_dir ./data/sharegpt.jsonl --output_dir ./data/llama/7b-5k-100k --model llama --datanum 5k
python sort_and_group.py --group_size 8 --train_file ./data/llama/7b-5k-100k
