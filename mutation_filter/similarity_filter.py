# coding: utf-8

import os
from transformers import AutoTokenizer, AutoModel
import torch
import json
from pprint import pprint

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)

def getSim(sent1,sent2):
    sentences = [sent1,sent2]
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return torch.cosine_similarity(sentence_embeddings[0],sentence_embeddings[1],dim=0)

with open("MG_JMV.json","r",encoding="utf-8") as f:
    mutations = json.load(f)

acc = 0
for ll,each in enumerate(mutations):
    premise,hypothesis = each['preSent']
    mutationList = each['mutationList']
    for index,item in enumerate(mutationList):
        mut1 = item['mut1']
        mut2 = item['mut2']
        pSim = getSim(premise,mut1)
        hSim = getSim(hypothesis,mut2)
        mutationList[index]['pSim'] = pSim.item()
        mutationList[index]['hSim'] = hSim.item()
    for index,item in enumerate(mutationList):
        assert item['pSim'] > 0 or item['pSim'] < 0
    if ll % 100 == 0:
        print(ll)
    each['mutationList'] = mutationList

with open("MG_JMV_SF.json","w",encoding="utf-8") as f:
    json_str = json.dumps(mutations)
    f.write(json_str)

