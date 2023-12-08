import torchvision.models as models
import math
import numpy as np
import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=112):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)



class TransformerPromptDecoder(nn.Module):
    def __init__(self, vocab_size=109, image_code_dim=768, num_encoder_layers=6,
                    num_decoder_layers=6, d_model=512, n_head=8, dim_feedforward=2048, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.image_code_embedding = nn.Linear(image_code_dim, d_model)
        self.sigmoid = nn.Sigmoid()
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)
        self.d_model_size = d_model

        self.decoder = nn.Transformer(d_model=d_model,nhead=n_head,num_encoder_layers=num_encoder_layers,
                                      num_decoder_layers=num_decoder_layers,
                                    dim_feedforward=dim_feedforward, batch_first=True)
        
        self.fc = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def get_key_padding_mask(self, tokens):
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 0] = -torch.inf
        return key_padding_mask

        
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.image_code_embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
        
    def predictor(self, out):
        out = self.fc(out)
        out = self.softmax(out)
        return out
    

    def forward(self, image_code, tgt):
        # 将图像网格表示转换为序列表示形式 
        batch_size, image_code_dim = image_code.size(0), image_code.size(1)
        # -> (batch_size, grid_height, grid_width, image_code_dim) 
        image_code = image_code.permute(0, 2, 3, 1)  
        # -> (batch_size, grid_height * grid_width, image_code_dim)
        image_code = image_code.view(batch_size, -1, image_code_dim)
        
        image_code = self.sigmoid(self.image_code_embedding(image_code))
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1], device=image_code.device)
        tgt_key_padding_mask = self.get_key_padding_mask(tgt).to(image_code.device)
        tgt = self.embedding(tgt.to(image_code.device))
        tgt = self.positional_encoding(tgt)
        
        out = self.decoder(image_code, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return out
        