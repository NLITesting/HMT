# Hybrid Mutation driven Testing for Natural Language Inference
This repository stores our experimental codes for the paper "Hybrid Mutation driven Testing for Natural Language Inference". HMT is short for the approach we proposed in this paper: **Hybrid Mutation driven Testing**.

## Datasets
We used two NLI datasets, MNLI and SNLI, which are placed in the dataset folder, and we mainly apply the test sets of the two datasets.

## Models
We used two state-of-the-art NLI models for testing, Decomposable Attention and RoBERTa. We call these two models through Python library allennlp and torch.

## Our Approach

Our method HMT comprises the following three steps:

(a) **Mutation Generation.** We design four mutation operators to replace the common nouns in the premise and the hypothesis sentences, where replacement words come from the Wordnet synonym set.

(b) **Mutation Filter.** Due to the existence of polysemy and the deviation of syntactic analysis, the mutations cannot guarantee high precision. We design a three-step filter to screen the mutations, including structural filter, similarity filter, and disambiguation filter.

(c) **Prediction Evaluation.** We import the mutated and original samples into the NLI model and mark the inconsistent predicted results as inconsistency bugs.

We place our code in three folders according to these three steps, and the folder has the same name as the step. The code in the mutation_generation folder is divided according to the data set. Each folder contains four .py files corresponding to four mutation operators. This part of the code includes mutation generation and structural filtering (the stanford-corenlp-4.4.0.zip file contains related models for structural filtering). The code of the similarity filter and disambiguation filter are placed in the mutation_filter folder. The code in the prediction_evaluation folder corresponds to the step prediction evaluation.

## Code for Discussion

We have organized parts of the codes from the discussion chapters of our article in the "code_for_discussion" folder.

The codes in the "Improvement_of_NLI_Models" folder correspond to Section 6.5 in the article. We place the finetuning code and data of the two models in the "Attention" and "RoBERTa" folders respectively.

The "Mutation_with_Other_Parts_of_Speech" folder corresponds to Section 6.6 in the article. We classify the codes according to the data set, namely MNLI and SNLI. We provide the code for the mutation generation part, and the code for the three-step filtering part is consistent with the main experiments. The codes ending with "JJ" represent the mutation generation of adjective substitution, and the codes ending with "VB" represent the mutation generation of verb substitution.
