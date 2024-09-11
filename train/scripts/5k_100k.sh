

deepspeed --include localhost:0,1,2,3,4,5,6,7 main.py \
--model_name_or_path THUDM/LongAlign-7B-64k-base \
--train_file ./data/llama/7b-5k-100k \
--output_dir ./output/llama/7b-5k-100k \
--num_train_epochs 2 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 12 \
--save_strategy "steps" \
--save_steps 200 \
--save_total_limit 10 \
--preprocessing_num_workers 64 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--logging_dir "./logs/" \
--deepspeed ds_config/stage3.json \
--bf16 \
--gradient_checkpointing 1 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--report_to "wandb" \
--run_name "7b-5k-100k" \
--logging_steps 1 \
--batch_method "sort" 