from model.ImagePromptGenerator import GridWithTransformer
from datasets.dataload import create_dataloader
import torch
import json

generator = GridWithTransformer().to('cpu')

generator.load_state_dict(torch.load('/workspace/image_prompt/sd-model-finetuned/checkpoint-5000/pytorch_model.bin'))


train_loader, val_loader, test_loader = create_dataloader(batch_size = 1, num_workers=2)

with open('vocab.json', 'r') as f:
    vocab = json.load(f)


new_dict={v:k for k,v in vocab.items()}
for i, (imgs, caps, caplens) in enumerate(train_loader):
    text = generator.generate_by_beamsearch(imgs.to('cpu'), 5, 102, vocab)
    print(text)
    for tx in text:
        for word in tx:
            print(new_dict[word], end=' ')
        print('\n')
    break