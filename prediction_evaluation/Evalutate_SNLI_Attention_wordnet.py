from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import json
from pprint import pprint
from allennlp.data import Instance
import os

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz")
indexMap = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}

def getPrediction(premise, hypothesis):
    res = predictor.predict(premise=premise, hypothesis=hypothesis)
    label_probs = res['label_probs']
    maxIndex = label_probs.index(max(label_probs))
    return label_probs,indexMap[maxIndex]

filename = 'MG_JMV_SF_DF'
with open(filename + ".json","r",encoding='utf-8') as f:
    mutations = json.load(f)

acc = 0
for index,each in enumerate(mutations):
    premise, hypothesis = each['preSent']
    target = each['target']
    mutationList = each['mutationList']
    label_probs,prediction = getPrediction(premise, hypothesis)
    each['prediction'] = prediction
    each['confidence'] = label_probs
    if prediction == target:
        acc += 1
    if len(each['mutationList']) == 0:
        continue
    for info in mutationList:
        mut1 = info['mut1']
        mut2 = info['mut2']
        info['confidence'],info['prediction'] = getPrediction(mut1, mut2)

    if index % 100 == 0:
        print(acc)

with open(filename + "_result_for_Attention.json", "w", encoding="utf-8") as f:
    json_str = json.dumps(mutations)
    f.write(json_str)