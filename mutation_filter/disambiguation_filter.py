import json
import random
from pprint import pprint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from pywsd.lesk import simple_lesk
from pywsd.lesk import cosine_lesk

label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
threshold = 0.9

filename = "MG_JMV_SF"
with open(filename+".json", "r", encoding="utf-8") as f:
    mutations = json.load(f)

new_mutations = []

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

for each in mutations:
    target = each['target']
    prediction = each['prediction']
    mutationsList = each['mutationList']
    mutationsList = [info for index, info in enumerate(mutationsList)
                        if info['pSim'] > threshold and info['hSim'] > threshold and info['passTest'] is True]

    tmpDict = {'preSent': each['preSent'], "target": each["target"], 'mutationList': []}

    for mut in mutationsList:
        answer1 = cosine_lesk(each["preSent"][0], mut["targetWord"], pos='n')
        answer2 = cosine_lesk(each["preSent"][1], mut["targetWord"], pos='n')
        answerm1 = cosine_lesk(mut['mut1'], mut["replaceWord1"], pos='n')
        answerm2 = cosine_lesk(mut['mut2'], mut["replaceWord2"], pos='n')
        if answer1 == answerm1 and answer2 == answerm2:
            tmpDict['mutationList'].append(mut)

    new_mutations.append(tmpDict)

import json
with open(filename + "_DF.json", "w", encoding="utf-8") as f:
    json_str = json.dumps(new_mutations)
    f.write(json_str)