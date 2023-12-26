# Standard Train and RL Train
##TODO


# img_inference
   if you need to use the specific model to inference specific image to get the text result, you could use the function in *img_predict.py*.

***A CASE TO USE img_predict TO GET RESULT***
```python
texts = img_predict(model_path, img_path, opt={})

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
