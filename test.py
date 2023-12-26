from model.ImagePromptGenerator import GridWithTransformer
from datasets.dataload import create_dataloader
from PIL import Image
import torch
import json
import torchvision.transforms as transforms
import sys
from metric.bleu import BLEUMetric
from argparse import Namespace

test_type = sys.argv[1] # img, bleu

with open('vocab.json', 'r') as f:
    vocab = json.load(f)

if test_type == 'img':
    model_path = sys.argv[2]    # pt file
    img_path = sys.argv[3]
    
    generator = GridWithTransformer(vision_encoder='Resnet').to('cuda')
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
    # text = generator.generate_by_beamsearch(img.to('cuda'), 5, 102, vocab)
    text, __ = generator.pre_sample(img.to('cuda'), 102, vocab)
    text = text.tolist()
    for tx in text:
        for word in tx:
            print(new_dict[word], end=' ')
        print('\n')

elif test_type == 'bleu':
    model_path = sys.argv[2]    # pt file
    generator = GridWithTransformer(vision_encoder="Resnet").to('cuda')
    generator.load_state_dict(torch.load(model_path))
    train_loader, val_loader, test_loader = create_dataloader(batch_size=256)
        # 设置模型超参数和辅助变量
    config = Namespace(
        max_len = 110,
        captions_per_image = 1,
        beam_k = 5
    )
    
    bleu4 = BLEUMetric.evaluate(test_loader, generator, config, vocab=vocab)
    
    with open('./bleu_score_RL_3000.json', 'w') as f:
        json.dump(
            {model_path:bleu4}, f
        )