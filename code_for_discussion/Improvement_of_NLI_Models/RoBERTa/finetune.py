from transformers import AutoTokenizer
import pandas as pd
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_input)

print(tokenizer.decode(encoded_input["input_ids"]))

train_dataset = pd.read_csv('retrain_for_Roberta.csv')
validation_dataset = pd.read_csv('retrain_validation_for_Roberta.csv')

from datasets import Dataset
ds_train = Dataset.from_dict(train_dataset)
ds_validation = Dataset.from_dict(validation_dataset)

def tokenize_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True)

tokenized_train_datasets = ds_train.map(tokenize_function, batched=True)
tokenized_validation_datasets = ds_validation.map(tokenize_function, batched=True)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli", num_labels=3)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="final_trainer", evaluation_strategy="epoch", num_train_epochs=3.0, load_best_model_at_end=True,
                                  save_strategy="epoch",per_device_train_batch_size=8)

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_validation_datasets,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()
torch.save(model, 'model.pt')
