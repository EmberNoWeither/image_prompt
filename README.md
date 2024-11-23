# It's just a homework code.
# Standard Train and RL Train
   we finished the RL Train and the score like CIDEr was truly improved but we found a strange situation that the model tends to say the same sentence and performed worse. :anguished:
   
   we used the ['Self-Critical Sequence Training'](https://arxiv.org/pdf/1612.00563) method to finish the part of RL and its based on [Wentong-DST's implementation](https://github.com/Wentong-DST/self-critical/tree/master).
# How to train?
You could run the code in 'dist_train.sh'.

```shell
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
```
Its default setting is standard train. 

If you want to run the RL train, please change the param of 'rl_after'.

Then it will run the RL train after the epoch which you choose.

# img_inference
   if you need to use the specific model to inference specific image to get the text result, you could use the function in *img_predict.py*.

***A CASE TO USE img_predict TO GET RESULT***
```python
from img_predict import img_predict

texts = img_predict(model_path='/checkpoint-15000/pytorch_model.bin',
         img_path='/data/deepfashion-multimodal/captions_img/MEN-Denim-id_00000182-01_7_additional.jpg', opt={})   # texts is a string value of results
print(texts)

#the upper clothing has no sleeves , cotton fabric and pure color patterns . it has a crew neckline . the lower clothing is of long length .
#the fabric is cotton and it has solid color patterns . there is an accessory on her wrist . there is a ring on her finger .

# a case for opt:
opt = {
    'predict_method': 'beam-k',  # the other method is 'random_sample'
    'k_value' : 5 ,  # a needed param for beam-k method
    'device' : 'cuda' ,  # you could also use 'cpu' to inference!
}
```
the param *"model_path"* is a string value to point the path for your model. 

the param *"img_path"* is a string value to point the path for your image to inference. 

the param *"opt"* could be the default value, we will use the beam-k method, and default k_value is 5. The inference will execute on cpu! 
