from pprint import pprint
import json
import torch
import os

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout for evaluation

label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}

def predict_pairs(sent1, sent2):
    tokens = roberta.encode(sent1, sent2)
    prediction = roberta.predict('mnli', tokens).argmax().item()
    confidence = roberta.predict('mnli', tokens).tolist()[0]
    prediction_label = label_map[prediction]
    return confidence, prediction_label

def evaluate(mutations):
    newMutationList = []
    cannotDealWith = 0
    for index, each in enumerate(mutations):
        newMutationDict = {}
        sent1, sent2 = each['preSent']
        newMutationDict['preSent'] = each['preSent']
        newMutationDict['target'] = each['target']
        pre_confidence, pre_label = predict_pairs(sent1, sent2)
        assert len(pre_confidence) == 3
        newMutationDict['prediction'] = pre_label
        newMutationDict['confidence'] = pre_confidence
        mutationInfo = []
        for mutes in each['mutationList']:
            muteDict = {}
            mut1, mut2 = mutes['mut1'], mutes['mut2']
            new_confidence, new_label = predict_pairs(mut1, mut2)
            muteDict['targetWord'] = mutes['targetWord']
            muteDict['replaceWord1'] = mutes['replaceWord1']
            muteDict['replaceWord2'] = mutes['replaceWord2']
            muteDict['mut1'] = mutes['mut1']
            muteDict['mut2'] = mutes['mut2']
            muteDict['passTest'] = mutes['passTest']
            muteDict['pSim'] = mutes['pSim']
            muteDict['hSim'] = mutes['hSim']
            muteDict['confidence'] = new_confidence
            mutationInfo.append(muteDict)
        newMutationDict['mutationList'] = mutationInfo
        if len(mutationInfo) == 0:
            cannotDealWith += 1
        newMutationList.append(newMutationDict)
        if index % 100 == 0:
            print(index)
    return newMutationList

if __name__ == "__main__":
    filename = 'MG_JMV_SF_DF'
    with open(filename + ".json", "r", encoding='utf-8') as f:
        mutations = json.load(f)
    mutationsAferEvaluate = evaluate(mutations)

    with open(filename + "_result_for_Attention.json", "w", encoding="utf-8") as f:
        json_str = json.dumps(mutationsAferEvaluate)
        f.write(json_str)
