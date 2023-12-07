from datasets.imgprompts import ImageTextDataset
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
import numpy as np
import json
import re


def create_dataloader(batch_size = 16, num_workers=8):
    train_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageTextDataset(
        '/workspace/image_prompt/datasets/train_data.json', '/workspace/image_prompt/vocab.json', 'train', transform=train_tx
    )
    
    valid_dataset = ImageTextDataset(
        '/workspace/image_prompt/datasets/test_data.json', '/workspace/image_prompt/vocab.json', 'val', transform=val_tx
    )

    test_dataset = ImageTextDataset(
        '/workspace/image_prompt/datasets/test_data.json', '/workspace/image_prompt/vocab.json', 'test', transform=val_tx
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader, test_loader


# train_loader, val_loader, test_loader = create_dataloader(batch_size = 16, num_workers=8)

# for i, (imgs, caps, caplens) in enumerate(train_loader):
#     print(caps.shape)
#     break

