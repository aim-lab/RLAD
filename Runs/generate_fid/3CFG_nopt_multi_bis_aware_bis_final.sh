#!/bin/bash
cd
source ../../oriondata/AIMLab/JonathanF/JonathanFNewFolder/environments/diffusion_env/bin/activate
cd ../../oriondata/AIMLab/JonathanF/JonathanFNewFolder/DiffusionDFI_repo
mkdir -p logs  # Use -p to prevent error if the directory already exists
mkdir -p logs/generate  # Use -p to prevent error if the directory already exists

export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# **Add dynamic MASTER_PORT assignment**
MASTER_PORT=$((10000 + RANDOM % 50000))
echo "Using MASTER_PORT: $MASTER_PORT"

export CUDA_LAUNCH_BLOCKING=1  # Set a value if needed
#--nnodes 1 --nproc_per_node 1 \
#--nnodes 1 --nproc_per_node 1 \
 #--master_port $MASTER_PORT \

ARGS="src/Difscripts/generate_new_data_fid.py \
--model_config_path src/configs/configsDiT/3ClassifierFreeGuidance_nopt_multi_bis_aware_final_AVnostrict.yaml \
--data_config_path src/configs/configsDiT/3ClassifierFreeGuidance_nopt_multi_bis_aware_final_AVnostrict.yaml \
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
--output_dir checkpoints/3ClassifierFreeGuidance_nopt_multi_bis_aware_final_AVnostrict \
"
#--deepspeed TransUNet_edited_vf/deepspeed/stage2.json

OMP_NUM_THREADS=2 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 nohup python $ARGS > "logs/generate/3ClassifierFreeGuidance_nopt_multi_bis_aware_final_lastAV_notStrict.txt" &
