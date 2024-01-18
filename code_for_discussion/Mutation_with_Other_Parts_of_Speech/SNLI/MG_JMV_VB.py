# coding: utf-8

from gensim.models import KeyedVectors
import json
from pprint import pprint
from nltk.corpus import wordnet as wn
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tmp_file = "../../gensim_word2vec.840B.300d.txt"

model = KeyedVectors.load_word2vec_format(tmp_file)

from stanfordcorenlp import StanfordCoreNLP
path = "../../stanford-corenlp-4.4.0"
nlp = StanfordCoreNLP(path,lang="en")
need = ['(VB ']



with open("../../data/SNLI/snli_1.0_test.jsonl","r",encoding="utf-8") as f:
    data = f.readlines()
    data = [json.loads(each) for each in data]

def getCandidate(line):
    sent1,sent2,target = line['sentence1'],line['sentence2'],line['gold_label']
    mutationDict = {'preSent': (sent1, sent2), 'target': target,
                    'mutationList': []}

    parse1, parse2 = nlp.parse(sent1), nlp.parse(sent2)

    target = []
    for word in need:
        while (True):
            if word in parse1:
                parse1, targetWord = getNoun(parse1, word)
                targetWord = targetWord.lower()
                target.append(targetWord)
            else:
                break

    parse1, parse2 = nlp.parse(sent1), nlp.parse(sent2)

    print(target)
    for targetWord in target:
        if targetWord not in sent2:
            continue
        try:
            sims = getSynset(targetWord)
            if len(sims) == 0:
                continue
        except:
            return mutationDict

        sim_list = []
        i_list = []
        j_list = []
        for i in range(len(sims) - 1):
            for j in range(i + 1, len(sims)):
                try:
                    similarity = model.similarity(sims[i], sims[j])
                except:
                    continue
                if checkSynset(sims[i], sims[j]):
                    sim_list.append(similarity)
                    i_list.append(i)
                    j_list.append(j)
        choose_list = np.argsort(sim_list)
        #if len(choose_list) > 20:
            #choose_list = choose_list[:20]

        for i in range(len(choose_list)):
            passTest = False
            mut1 = sent1.replace(targetWord, sims[i_list[choose_list[i]]])
            mut2 = sent2.replace(targetWord, sims[j_list[choose_list[i]]])
            try:
                parseMut1 = nlp.parse(mut1)
                parseMut2 = nlp.parse(mut2)
                if parseMut1.replace(sims[i_list[choose_list[i]]], targetWord) == parse1 and parseMut2.replace(
                        sims[j_list[choose_list[i]]], targetWord) == parse2:
                    passTest = True
                tmpDict = {'targetWord': targetWord, 'replaceWord1': sims[i_list[choose_list[i]]],
                           'replaceWord2': sims[j_list[choose_list[i]]],
                           'mut1': mut1, 'mut2': mut2, 'passTest': passTest}
                mutationDict['mutationList'].append(tmpDict)
            except:
                tmpDict = {'targetWord': targetWord, 'replaceWord1': sims[i_list[choose_list[i]]],
                           'replaceWord2': sims[j_list[choose_list[i]]],
                           'mut1': mut1, 'mut2': mut2, 'passTest': passTest}
                mutationDict['mutationList'].append(tmpDict)

            passTest = False
            mut1 = sent1.replace(targetWord, sims[j_list[choose_list[i]]])
            mut2 = sent2.replace(targetWord, sims[i_list[choose_list[i]]])
            try:
                parseMut1 = nlp.parse(mut1)
                parseMut2 = nlp.parse(mut2)
                if parseMut1.replace(sims[j_list[choose_list[i]]], targetWord) == parse1 and parseMut2.replace(
                        sims[i_list[choose_list[i]]],
                        targetWord) == parse2:
                    passTest = True
                tmpDict = {'targetWord': targetWord, 'replaceWord1': sims[j_list[choose_list[i]]],
                           'replaceWord2': sims[i_list[choose_list[i]]],
                           'mut1': mut1, 'mut2': mut2, 'passTest': passTest}
                mutationDict['mutationList'].append(tmpDict)
            except:
                tmpDict = {'targetWord': targetWord, 'replaceWord1': sims[j_list[choose_list[i]]],
                           'replaceWord2': sims[i_list[choose_list[i]]],
                           'mut1': mut1, 'mut2': mut2, 'passTest': passTest}
                mutationDict['mutationList'].append(tmpDict)
    return mutationDict


def getNoun(parse, word):
    startIndex = parse.index(word)
    startIndex1 = startIndex + len(word)
    parse = parse[:startIndex] + '(OK ' + parse[startIndex1:]
    lastIndex = startIndex1
    while parse[lastIndex] != ')':
        lastIndex += 1
    return parse, parse[startIndex1:lastIndex]


def getSynset(targetWord):
    set = wn.synsets(targetWord, pos=wn.VERB)
    synset = []
    for w in set:
        wset = w.lemma_names()
        for i in wset:
            if i in synset:
                continue
            if i == targetWord:
                continue
            else:
                synset.append(i)
    return synset


def checkSynset(targetWord1, targetWord2):
    set = wn.synsets(targetWord1, pos=wn.VERB)
    synset = []
    for w in set:
        # print(w.lemma_names())
        wset = w.lemma_names()
        for i in wset:
            if i in synset:
                continue
            if i == targetWord1:
                continue
            else:
                synset.append(i)
    if targetWord2 in synset:
        return True
    else:
        return False

import datetime
starttime = datetime.datetime.now()

mutations = []
examCount = 0
for index, line in enumerate(data):
        mutationDict = getCandidate(line)
        mutations.append(mutationDict)
        examCount += len(mutationDict['mutationList'])
        if index % 100 == 0:
            print(index)
            print("mutation number:", examCount)

print("mutation number:", examCount)
endtime = datetime.datetime.now()
timeuse = (endtime - starttime).seconds
print(timeuse)

with open("MG_JMV_VB.json", "w", encoding="utf-8") as f:
    json_str = json.dumps(mutations)
    f.write(json_str)
    


