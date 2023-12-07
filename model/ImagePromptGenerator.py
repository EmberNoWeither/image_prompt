import torchvision.models as models
import math
import numpy as np
import torch.nn as nn
import torch
from model.VisionEncoder import ViTEncoder
from model.PromptDecoder import TransformerPromptDecoder
from losses.loss import PackedCrossEntropyLoss


class GridWithTransformer(nn.Module):
    def __init__(self, vocab_size=109, image_code_dim=768, num_encoder_layers=6,
                    num_decoder_layers=6, d_model=512, n_head=8, 
                    dim_feedforward=2048,image_size=256,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = ViTEncoder(image_size=image_size)
        self.decoder = TransformerPromptDecoder(vocab_size=vocab_size, image_code_dim=image_code_dim, num_encoder_layers=num_encoder_layers,
                                                    num_decoder_layers=num_decoder_layers, d_model=d_model, n_head=n_head, 
                                                    dim_feedforward=dim_feedforward)
        self.loss_fn = PackedCrossEntropyLoss()
        
    
    def train_step(self, imgs, captions, caplens):
        sorted_cap_lens, sorted_cap_indices = torch.sort(caplens, 0, True)
        captions = captions[sorted_cap_indices]
        
        image_code = self.encoder(imgs)
        image_code = image_code[sorted_cap_indices]
        
        sorted_cap_lens = sorted_cap_lens.cpu().numpy() - 1     # 序列-1 最后一个时刻不需要预测下一个词
        
        
        outs = self.decoder(image_code.to('cuda'), captions.to('cuda'))
        preds = self.decoder.predictor(outs)
        
        log_var = {}
        loss = self.loss_fn(preds, captions[:, 1:], sorted_cap_lens)
        
        log_var.update(loss_kpt = loss)
        return log_var, loss
        
        
    def predict_step(self, imgs, caps, caplens):
        pass
        