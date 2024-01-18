import json

import torch
from allennlp.data import DatasetReader, Instance, Vocabulary, TextFieldTensors
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp_models.pair_classification.dataset_readers import SnliReader
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

config = {
  "dataset_reader": {
    "type": "snli",
    "token_indexers": {
      "elmo": {
        "type": "elmo_characters"
      }
    },
    "tokenizer": {
      "end_tokens": ["@@NULL@@"]
    }
  },
  "train_data_path": "train.jsonl",
  "validation_data_path": "valid.jsonl",
  "model": {
    "type": "from_archive",
    "archive_file": "decomposable-attention-elmo-2020.04.09.tar.gz",
   },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 64
    }
  },
  "trainer": {
    "num_epochs": 5,
    "patience": 20,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    },
    "cuda_device": 0
  }
}
config_filename = "training_config.json"
with open(config_filename, "w") as config_file:
    json.dump(config, config_file)
from allennlp.commands.train import train_model_from_file

train_model_from_file(
    config_filename, 'train_att', file_friendly_logging=True, force=True
)