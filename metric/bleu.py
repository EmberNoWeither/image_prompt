
from nltk.translate.bleu_score import corpus_bleu
import torch.nn as nn
import torch


def filter_useless_words(sent, filterd_words):
# 去除句子中不参与BLEU值计算的符号
    return [w for w in sent if w not in filterd_words]


class BLEUMetric(nn.Module):
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
                # 候选文本
                cands.extend([filter_useless_words(text, filterd_words) for text in texts])
                # # 参考文本
                refs.extend([filter_useless_words(cap, filterd_words) for cap in caps.tolist()])
                
                
        # 实际上，每个候选文本对应cpi条参考文本
        multiple_refs = []
        for idx in range(len(refs)):
            multiple_refs.append(refs[(idx//cpi)*cpi : (idx//cpi)*cpi+cpi])
        # 计算BLEU-4值，corpus_bleu函数默认weights权重为(0.25,0.25,0.25,0.25)
        # 即计算1-gram到4-gram的BLEU几何平均值
        bleu4 = corpus_bleu(multiple_refs, cands, weights=(0.25,0.25,0.25,0.25))
        model.train()
        return bleu4