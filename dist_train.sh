export CUDA_VISIBLE_DEVICES=4,5,6,7
# export ALL_PROXY='127.0.0.1:7890'

accelerate launch --main_process_port=25645 --num_processes=4 --mixed_precision="fp16" train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_batch_size=16 \
  --dataloader_num_workers=32 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=200000 \
  --checkpointing_steps=20000 \
  --learning_rate=1e-5 \
#   --use_8bit_adam \
  --output_dir="/workspace/image_prompt/checkpoints/deepfashion_GT" 
