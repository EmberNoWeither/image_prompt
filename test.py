from model.ImagePromptGenerator import GridWithTransformer
from datasets.dataload import create_dataloader
import torch

generator = GridWithTransformer().to('cuda')

generator.load_state_dict(torch.load('/workspace/image_prompt/sd-model-finetuned/checkpoint-5000/pytorch_model.bin'))


train_loader, val_loader, test_loader = create_dataloader(batch_size = 16, num_workers=8)

for i, (imgs, caps, caplens) in enumerate(val_loader):
    log_var, loss = generator.train_step(imgs.to('cuda'), caps.to('cuda'), caplens.to('cuda'))
    print(loss)
    break