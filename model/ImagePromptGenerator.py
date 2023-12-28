import torchvision.models as models
import json
import numpy as np
import torch.nn as nn
import torch
from model.VisionEncoder import ViTEncoder, ResnetEncoder
from model.PromptDecoder import TransformerPromptDecoder
from losses.loss import PackedCrossEntropyLoss, RewardCriterion
from metric.bleu import filter_useless_words
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import numpy as np


class GridWithTransformer(nn.Module):
    def __init__(self, vocab_size=109, image_code_dim=1000, num_encoder_layers=3,
                    num_decoder_layers=3, d_model=512, n_head=8, 
                    dim_feedforward=2048,image_size=224, vision_encoder = 'ViT',*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        if vision_encoder == 'ViT':
            self.encoder = ViTEncoder(image_size=image_size)
        elif vision_encoder == 'Resnet':
            self.encoder = ResnetEncoder()
            # image_code_dim = 2048
            
        self.decoder = TransformerPromptDecoder(vocab_size=vocab_size, image_code_dim=image_code_dim, num_encoder_layers=num_encoder_layers,
                                                    num_decoder_layers=num_decoder_layers, d_model=d_model, n_head=n_head, 
                                                    dim_feedforward=dim_feedforward)
        self.loss_fn = PackedCrossEntropyLoss()
        self.rl_fn = RewardCriterion()
        
        with open('vocab.json', 'r') as f:
            vocab = json.load(f)
        
        self.vocab = vocab
        
        self.bleu_weight = 0.7
        self.meteor_weight = 0.3
        
    
    def train_step(self, imgs, captions, caplens):
        sorted_cap_lens, sorted_cap_indices = torch.sort(caplens, 0, True)
        captions = captions[sorted_cap_indices]
        
        image_code = self.encoder(imgs)
        image_code = image_code[sorted_cap_indices]
        
        sorted_cap_lens = sorted_cap_lens.cpu().numpy() - 1     # 序列-1 最后一个时刻不需要预测下一个词
        
        preds = self.decoder(image_code.to('cuda'), captions.to('cuda'))
        # preds = self.decoder.predictor(outs)
        
        log_var = {}

        loss = self.loss_fn(preds, captions[:, 1:], sorted_cap_lens)
        
        log_var.update(loss_kpt = loss)
        return log_var, loss
    
    
    def rl_train_step(self, imgs, captions, caplens):
        vocab = self.vocab
        # 存储候选文本
        greedy_cands = []
        gen_cands = []
        # 存储参考文本
        refs = []
        # 需要过滤的词
        filterd_words = set({vocab['<start>'], vocab['<end>'], vocab['<pad>']})
        cpi = 1
        device = next(self.parameters()).device
        
        gen_result, sampleLogprob = self.pre_sample(imgs, 110, vocab)
        gen_cands.extend([filter_useless_words(text, filterd_words) for text in gen_result.tolist()])
        with torch.no_grad():
            greedy_texts = self.generate_by_beamsearch(imgs.to(device), 1, 110, vocab)
            # 候选文本
            greedy_cands.extend([filter_useless_words(text, filterd_words) for text in greedy_texts])
            # # 参考文本
            refs.extend([filter_useless_words(cap, filterd_words) for cap in captions.tolist()])
                
                
        multiple_refs = []
        for idx in range(len(refs)):
            multiple_refs.append(refs[(idx//cpi)*cpi : (idx//cpi)*cpi+cpi])
            
        refs = [[str(word) for word in subrefs] for subrefs in refs]
        gen_cands = [[str(word) for word in subrefs] for subrefs in gen_cands]
        greedy_cands = [[str(word) for word in subrefs] for subrefs in greedy_cands]

        #bleu
        gen_bleu4 = np.array([sentence_bleu([ref],cand) for ref,cand in zip(refs,gen_cands)])
        greedy_bleu4 = np.array([sentence_bleu([ref],cand) for ref,cand in zip(refs,greedy_cands)])

        # meteor
        gen_meteor = [meteor_score([ref],cand) for ref,cand in zip(refs,gen_cands)]
        gen_meteor_score = np.array(gen_meteor)
        
        greedy_meteor = [meteor_score([ref],cand) for ref,cand in zip(refs,greedy_cands)]
        greedy_meteor_score = np.array(greedy_meteor)
                
        gen_score = self.bleu_weight * gen_bleu4 + self.meteor_weight * gen_meteor_score
        greedy_score = self.bleu_weight * greedy_bleu4 + self.meteor_weight * greedy_meteor_score

        scores = gen_score - greedy_score
        rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
        
        loss = self.rl_fn(sampleLogprob, gen_result, torch.from_numpy(rewards).float().cuda())
        log_var = {}
        
        log_var.update(loss_kpt = loss)
        
        return log_var, loss
    
    
    def pre_sample(self, images, max_len, vocab):
        vocab_size = len(vocab)
        image_codes = self.encoder(images)
        texts = []
        device = images.device
        
        batch_size = image_codes.size(0)
        
        seq = image_codes.new_zeros(batch_size, max_len+2, dtype=torch.long)
        seqLogprobs = image_codes.new_zeros(batch_size, max_len+2)
        
        cur_sents = torch.full((batch_size, 1,), vocab['<start>'], dtype=torch.long).to(device)

        for t in range(max_len + 1):
            if cur_sents.size(1) >= max_len+2:
                break
            preds = self.decoder(image_codes, cur_sents)[:, -1]
            # -> (k, vocab_size)
            preds = self.decoder.predictor(preds)

            prob_prev = torch.exp(preds.data).cpu()

            it = torch.multinomial(prob_prev, 1).cuda()
            sampleLogprobs = preds.gather(1, it)

            it = it.view(-1).long()
            # stop when all finished
            if t == 0:
                unfinished = (it != vocab['<end>'])
            else:
                unfinished = unfinished * (it != vocab['<end>'])

            temp = unfinished.long()
            it = it * temp

            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            
            cur_sents = torch.cat([cur_sents, it.view(batch_size, -1)], dim=1)

            if unfinished.sum() == 0:
                # print(seq.tolist())
                break
            
        return seq, seqLogprobs
          
        
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
                    values, indices = probs.view(-1).topk(k, 0, True, True)

                # 计算最大的k个值对应的句子索引和词索引
                sent_indices = torch.div(indices, vocab_size, rounding_mode='trunc')
                word_indices = indices % vocab_size 
                
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
            
            # print(len(gen_sent))
            texts.append(gen_sent)
        return texts
    
    def predict_step(self, imgs, caps, caplens):
        pass
        