from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import re
import json
from metric.evaluate import evaluate

with open('/workspace/image_prompt/coco_karpathy_test.json', 'r') as f:
    coco_format_ann = json.load(f)
    

with open('/workspace/image_prompt/test_epoch0.json', 'r') as f:
    test_result = json.load(f)
    

with open('./vocab.json', 'r') as f:
    vocab = json.load(f)
    

refs = []
cans = []

for res in test_result:
    test_id = res['image_id']
    for ann in coco_format_ann:
        if test_id == int(ann['image_id']):
            test_cap = res['caption']
            test_cap = "".join(test_cap)
            test_cap = re.sub(r'([.,!?])',r' \1 ',test_cap)
            test_cap = re.sub(r'[^a-zA-Z.,!?]+',r' ',test_cap)
            tokens = test_cap.lower().split()
            enc_c = [vocab['<start>']] + [vocab.get(word, vocab['<unk>']) for word in tokens] + [vocab['<end>']]
            cans.append(enc_c)
            
            caption = "".join(ann['caption'])
            caption = re.sub(r'([.,!?])',r' \1 ',caption)
            caption = re.sub(r'[^a-zA-Z.,!?]+',r' ',caption)
            tokens = caption.lower().split()
            enc_c = [vocab['<start>']] + [vocab.get(word, vocab['<unk>']) for word in tokens] + [vocab['<end>']]
            refs.append(enc_c)

print(len(refs))
print(len(cans))

print(evaluate(refs, cans, vocab))