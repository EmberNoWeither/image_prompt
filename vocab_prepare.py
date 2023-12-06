import os
import re
from os.path import join as pjoin
import sys
import json
from collections import defaultdict, Counter
from PIL import Image
from matplotlib import pyplot as plt

vocab_folder = './'
origin_datasets = sys.argv[1] # json文件

vocab = Counter()

with open(origin_datasets, 'r') as f:
    train_sets = json.load(f)


for img, caption in zip(list(train_sets.keys()), list(train_sets.values())):
    caption = re.sub(r'([.,!?])',r' \1 ',caption)
    caption = re.sub(r'[^a-zA-Z.,!?]+',r' ',caption)
    tokens = caption.lower().split()
    vocab.update(tokens)


words = [w for w in vocab.keys()]
vocab = {k: v + 1 for v, k in enumerate(words)}
vocab['<pad>'] = 0
vocab['<unk>'] = len(vocab)
vocab['<start>'] = len(vocab)
vocab['<end>'] = len(vocab)


# 存储词典
with open(os.path.join(vocab_folder, 'vocab.json'), 'w') as fw:
    json.dump(vocab, fw)
       
dataset_folder = './datasets'
image_folder = '/workspace/data/deepfashion-multimodal/captions_img'

origin_data = [origin_datasets, sys.argv[2]]    # Train, Test Sets

for idx, file in enumerate(origin_data):
    with open(file, 'r') as f:
        origin_sets = json.load(f)
        enc_captions = []
        img_paths = []
        for img, caption in zip(list(origin_sets.keys()), list(origin_sets.values())):
            caption = re.sub(r'([.,!?])',r' \1 ',caption)
            caption = re.sub(r'[^a-zA-Z.,!?]+',r' ',caption)
            tokens = caption.lower().split()
            enc_c = [vocab['<start>']] + [vocab.get(word, vocab['<unk>']) for word in tokens] + [vocab['<end>']]
            img_paths.append(pjoin(image_folder, img))
            enc_captions.append(enc_c)

        if idx == 0:    # Trainset
            # 存储数据
            data = {'IMAGES': img_paths, 
                    'CAPTIONS': enc_captions}
            with open(os.path.join(dataset_folder, 'train' + '_data.json'), 'w') as fw:
                json.dump(data, fw)
        elif idx == 1:    # Testset
            data = {'IMAGES': img_paths, 
                    'CAPTIONS': enc_captions}
            with open(os.path.join(dataset_folder, 'test' + '_data.json'), 'w') as fw:
                json.dump(data, fw)