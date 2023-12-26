# Standard Train and RL Train
##TODO


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
