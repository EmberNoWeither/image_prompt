export CUDA_VISIBLE_DEVICES=0,1,2
# export ALL_PROXY='127.0.0.1:7890'

accelerate launch --main_process_port=25555 --num_processes=3 --mixed_precision="fp16" train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=32 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=3000 \
  --checkpointing_steps=500 \
  --learning_rate=5e-5 \
  --rl_after=-1 \
#   --use_8bit_adam \
  --output_dir="/workspace/image_prompt/checkpoints/deepfashion_GT" 
