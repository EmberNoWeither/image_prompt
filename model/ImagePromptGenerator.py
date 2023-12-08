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
        
    @torch.no_grad()
    def generate_by_beamsearch(self, images, beam_k, max_len, vocab):
        vocab_size = len(vocab)
        image_codes = self.encoder(images)
        texts = []
        device = images.device
        # 对每个图像样本执行束搜索
        for image_code in image_codes:
            # 将图像表示复制k份
            image_code = image_code.unsqueeze(0).repeat(beam_k,1,1,1)
            
            # 生成k个候选句子，初始时，仅包含开始符号<start>
            cur_sents = torch.full((beam_k, 1), vocab['<start>'], dtype=torch.long).to(device)

            sent_lens = torch.LongTensor([1]*beam_k).to(device)
            
            # 存储已生成完整的句子（以句子结束符<end>结尾的句子）
            end_sents = []
            # 存储已生成完整的句子的概率
            end_probs = []
            # 存储未完整生成的句子的概率
            probs = torch.zeros(beam_k, 1).to(device)
            k = beam_k
            while True:
                preds = self.decoder(image_code[:k], cur_sents)[:, -1]
                # -> (k, vocab_size)
                preds = self.decoder.predictor(preds).view(k, -1)
                
                # 对每个候选句子采样概率值最大的前k个单词生成k个新的候选句子，并计算概率
                # -> (k, vocab_size)
                probs = probs.repeat(1,preds.size(1)) + preds
                if cur_sents.size(1) == 1:
                    # 第一步时，所有句子都只包含开始标识符，因此，仅利用其中一个句子计算topk
                    values, indices = probs[0].topk(k, 0, True, True)
                else:
                    # probs: (k, vocab_size) 是二维张量
                    # topk函数直接应用于二维张量会按照指定维度取最大值，这里需要在全局取最大值
                    # 因此，将probs转换为一维张量，再使用topk函数获取最大的k个值
                    # print(probs.shape)
                    values, indices = probs.view(-1).topk(k, 0, True, True)
                    # print(indices)
                # 计算最大的k个值对应的句子索引和词索引
                sent_indices = torch.div(indices, vocab_size, rounding_mode='trunc')
                # print(sent_indices)
                word_indices = indices % vocab_size 
                
                # print(cur_sents[sent_indices].shape)
                # print(word_indices.unsqueeze(-1).shape)

                # 将词拼接在前一轮的句子后，获得此轮的句子
                cur_sents = torch.cat([cur_sents[sent_indices], word_indices.unsqueeze(1)], dim=1)
                # 查找此轮生成句子结束符<end>的句子
                end_indices = [idx for idx, word in enumerate(word_indices) if word.detach().item() == vocab['<end>']]
                    
                if len(end_indices) > 0:
                    end_probs.extend(values[end_indices])
                    end_sents.extend(cur_sents[end_indices].tolist())
                    # 如果所有的句子都包含结束符，则停止生成
                    k -= len(end_indices)
                    if k == 0:
                        break
                # 查找还需要继续生成词的句子
                cur_indices = [idx for idx, word in enumerate(word_indices) 
                               if word != vocab['<end>']]
                if len(cur_indices) > 0:
                    cur_sent_indices = sent_indices[cur_indices]
                    cur_word_indices = word_indices[cur_indices]
                    # 仅保留还需要继续生成的句子、句子概率、隐状态、词嵌入
                    cur_sents = cur_sents[cur_indices]
                    probs = values[cur_indices].view(-1,1)
                # 句子太长，停止生成
                if cur_sents.size(1) >= max_len:
                    break
            if len(end_sents) == 0:
                # 如果没有包含结束符的句子，则选取第一个句子作为生成句子
                gen_sent = cur_sents[0].tolist()
            else: 
                # 否则选取包含结束符的句子中概率最大的句子
                gen_sent = end_sents[end_probs.index(max(end_probs))]
            texts.append(gen_sent)
            print(texts)
        return texts
    
    def predict_step(self, imgs, caps, caplens):
        pass
        