from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import torch
import torch.nn as nn
from metric.bleu import filter_useless_words


def evaluate(refs, cands, vocab):
    idx2word = {value: key for key, value in vocab.items()}
    
    refs = [[idx2word.get(word, word) for word in sublist] for sublist in refs]
    cands = [[idx2word.get(word, word) for word in sublist] for sublist in cands]
    
    # meteor
    scores = [meteor_score([ref],cand) for ref,cand in zip(refs,cands)]
    avg_score = sum(scores)/len(scores)

    # 字符列表转字符串给rouge
    refs_str = [' '.join(sublist) for sublist in refs]
    cands_str = [' '.join(sublist) for sublist in cands]
   
    # 初始化空列表用于存放修改后的参考文本和生成文本
    filtered_refs = []
    filtered_cands = []

    # 遍历原始列表，将生成文本为空的对应参考文本删除
    for ref, cand in zip(refs_str, cands_str):
        if cand:
            filtered_refs.append(ref)
            filtered_cands.append(cand)
            
    # rouge
    rouge = Rouge()
    scores_r = rouge.get_scores(filtered_cands,filtered_refs,avg=True)
    rouge_2 = scores_r['rouge-2']['r']
    rouge_L = scores_r['rouge-l']['r']
    
    # 将 single_ref 设置为真实的参考文本列表
    single_refs = [[ref] for ref in refs]        
    
    # BLEU
    bleu4 = corpus_bleu(single_refs, cands, weights=(0.25,0.25,0.25,0.25))
    
    return bleu4,avg_score,rouge_2,rouge_L


class EvaluateMetric(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod 
    @torch.no_grad()
    def evaluate(data_loader, model, config, vocab):
        model.eval()
        # 存储候选文本
        cands = []
        # 存储参考文本
        refs = []
        # 需要过滤的词
        filterd_words = set({vocab['<start>'], vocab['<end>'], vocab['<pad>']})
        cpi = config.captions_per_image
        device = next(model.parameters()).device
        for i, (imgs, caps, caplens) in enumerate(data_loader):
            with torch.no_grad():
                # 通过束搜索，生成候选文本
                texts = model.generate_by_beamsearch(imgs.to(device), config.beam_k, config.max_len+2, vocab)
                # texts, __ = model.pre_sample(imgs.to(device),  config.max_len+2, vocab)
                # texts = texts.tolist()
                # 候选文本
                cands.extend([filter_useless_words(text, filterd_words) for text in texts])
                # # 参考文本
                refs.extend([filter_useless_words(cap, filterd_words) for cap in caps.tolist()])
                
        bleu_score, meteor_score, rouge_2_score, rouge_L_score = evaluate(refs, cands, vocab)
        
        return bleu_score, meteor_score, rouge_2_score, rouge_L_score