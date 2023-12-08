import torchvision.models as models
import torch.nn as nn
import torch

# 该最终forward后网格表示为(768,x,x)的特征张量
class ViTEncoder(nn.Module):
    def __init__(self, image_size=256,finetuned=True,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model = models.vit_b_32(image_size=image_size)
        
        self.grid_rep_extractor = nn.Sequential(*(list(model.children())[:-2]))
        for param in self.grid_rep_extractor.parameters():
            param.requires_grad = finetuned
            
    def forward(self, images):
        out = self.grid_rep_extractor(images) 
        return out
    

# 该最终forward后网格表示为(2048,x,x)的特征张量
class ResnetEncoder(nn.Module):
    def __init__(self, image_size=256,finetuned=True,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model = models.resnet152()
        
        self.grid_rep_extractor = nn.Sequential(*(list(model.children())[:-2]))
        for param in self.grid_rep_extractor.parameters():
            param.requires_grad = finetuned
            
    def forward(self, images):
        out = self.grid_rep_extractor(images) 
        return out
    