#!/bin/bash
cd ../
source RLAD_env/bin/activate
mkdir -p logs  # Use -p to prevent error if the directory already exists
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

export CUDA_LAUNCH_BLOCKING=1  # Set a value if needed

ARGS="Scripts/generate_new_data.py \
--model_config_path configs/configsDiT/RLAD.yaml \
--do_train True \
--dataloader_num_workers 30 \
--per_device_eval_batch_size 16 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--warmup_steps 1000 \
--learning_rate 1e-4 \
--lr_scheduler_type cosine \
--weight_decay 0.05 \
--num_train_epochs 10 \
--save_total_limit 20 \
--bf16 True \
--push_to_hub=False \
--save_strategy epoch \
--load_best_model_at_end True \
--output_dir checkpoints/RLAD \
"

OMP_NUM_THREADS=2 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 nohup python $ARGS > "logs/RLAD_generate.txt" &

