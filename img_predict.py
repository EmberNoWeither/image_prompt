from model.ImagePromptGenerator import GridWithTransformer
from datasets.dataload import create_dataloader
from PIL import Image
import torch
import json
import torchvision.transforms as transforms
import sys
from metric.bleu import filter_useless_words
from argparse import Namespace

"""
opt = {
    'predict_method': 'beam-k' or 'random_sample'
    'k_value' : 5 (a param for beam-k),
    'device' : 'cuda' or 'cpu',
}
"""
def img_predict(model_path, img_path, opt={}):
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
        
    method = opt.get('predict_method', 'beam-k')
    device = opt.get('device', 'cpu')
    k = opt.get('k_value', 5)
    
    generator = GridWithTransformer(vision_encoder='Resnet').to(device)
    generator.load_state_dict(torch.load(model_path))

    val_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')

    img = val_tx(img).reshape((1,3,224,224))
    new_dict={v:k for k,v in vocab.items()}
    
    if method == 'beam-k':
        texts = generator.generate_by_beamsearch(img.to(device), k, 102, vocab)
    elif method == 'random_sample':
        texts, __ = generator.pre_sample(img.to(device), 102, vocab)
        texts = texts.tolist()
    
    filterd_words = set({vocab['<start>'], vocab['<end>'], vocab['<pad>']})
    text_show = []
    text_show.extend([filter_useless_words(text, filterd_words) for text in texts])
        
    for tx in text_show:
        for word in tx:
            print(new_dict[word], end=' ')
        print('\n')
        
    text = [new_dict[wd]+ " " for wd in tx for tx in text_show]
    text = "".join(text)
        
    return text


# pre_opt = {
#     'device':'cuda',
#     # 'predict_method' : 'random_sample'
# }

# text = img_predict('/workspace/image_prompt/sd-model-finetuned-base/checkpoint-15000/pytorch_model.bin', '/workspace/data/deepfashion-multimodal/captions_img/MEN-Denim-id_00000182-01_7_additional.jpg', pre_opt)

# print(text)