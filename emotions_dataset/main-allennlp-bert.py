from operator import mod
import tempfile
from typing import Dict, Iterable, List, Tuple

import torch

from allennlp.common.util import JsonDict
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.predictors import Predictor
from allennlp.training import Trainer, GradientDescentTrainer
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.util import evaluate
from allennlp.predictors.text_classifier import TextClassifierPredictor
from allennlp.interpret.saliency_interpreters import SimpleGradient
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler
import numpy as np
import matplotlib.pyplot as plt

import LeavenoutViz
import LeaveoneoutViz
import SaliencymapViz


# There's a warning when you call `forward_on_instances` that you don't need
# to worry about right now, so we silence it.
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

highlight_color_1 = (0, 0.5, 1.0)
highlight_color_2 = (0, 1.0, 0)

# Change to True to train model
TRAIN = False

class ClassificationTsvReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name = "bert-base-uncased")
        self.token_indexers = token_indexers or {"bert": PretrainedTransformerIndexer(model_name = "bert-base-uncased")}
        self.max_tokens = max_tokens

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        df = pd.read_csv(file_path, sep=',', index_col=0, header=0)

        for index, row in df.iterrows():
            tokens = self.tokenizer.tokenize(row["DATA_COLUMN"])
            if self.max_tokens:
                    tokens = tokens[: self.max_tokens]

            text_field = TextField(tokens, self.token_indexers)
            label_field = LabelField(str(row["LABEL_COLUMN"]))
            fields: Dict[str, Field] = {"text": text_field, "label": label_field}

            #print(row["LABEL_COLUMN"], row["DATA_COLUMN"])
            yield Instance(fields)

class SimpleClassifier(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder, use_softmax = True
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.use_softmax = use_softmax

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)

        if self.use_softmax:
            # Shape: (batch_size, num_labels)
            probs = torch.nn.functional.softmax(logits)
        else:
            probs = logits

        output = {'probs': probs}
        if label is not None:
            self.accuracy(logits, label)
            # Shape: (1,)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(TextClassifierPredictor):
    def predict(self, sentence: str) -> JsonDict:
        # This method is implemented in the base class.
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)

def build_dataset_reader() -> DatasetReader:
    return ClassificationTsvReader()


def read_data(reader: DatasetReader) -> Tuple[List[Instance], List[Instance]]:
    print("Reading data")
    training_data = list(reader.read("dataset/merged_training_train.csv"))
    validation_data = list(reader.read("dataset/merged_training_test.csv"))
    return training_data, validation_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary, use_softmax = True) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"bert": PretrainedTransformerEmbedder(model_name = "bert-base-uncased", train_parameters = False)}
    )
    encoder = BertPooler(pretrained_model = "bert-base-uncased", requires_grad = False)
    return SimpleClassifier(vocab, embedder, encoder, use_softmax)

def run_training_loop():
    dataset_reader = build_dataset_reader()

    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab).to(torch.cuda.current_device())

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
        print("Starting training")
        trainer.train()
        print("Finished training")

    return model, dataset_reader, vocab


def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, 64, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, 64, shuffle=False)
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr=0.00005)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=30,
        optimizer=optimizer,
        patience = 2,
        cuda_device=0
    )
    return trainer

if TRAIN:
    model, dataset_reader, vocab = run_training_loop()

    model = model.cpu()

    with open("model-bert.th", 'wb') as f:
        torch.save(model.state_dict(), f)

    vocab.save_to_files("bert_vocabulary")

    # Now we can evaluate the model on a new dataset.
    test_data = list(dataset_reader.read("dataset/merged_training_test.csv"))
    data_loader = SimpleDataLoader(test_data, batch_size=128)
    data_loader.index_with(model.vocab)

    model = model.to(torch.cuda.current_device())
    results = evaluate(model, data_loader, cuda_device = torch.cuda.current_device())
    print(results)    

else:

    vocab = Vocabulary.from_files("bert_vocabulary")
    model = build_model(vocab, True)
    dataset_reader = build_dataset_reader()

    with open("model-bert.th", 'rb') as f:
        model.load_state_dict(torch.load(f))
        

    # test_data = list(dataset_reader.read("dataset/merged_training_test.csv"))
    # data_loader = SimpleDataLoader(test_data, batch_size=128)
    # data_loader.index_with(model.vocab)

    model = model.to(torch.cuda.current_device())
    # results = evaluate(model, data_loader, cuda_device = torch.cuda.current_device())
    # print(results)


    predictor = SentenceClassifierPredictor(model, dataset_reader)

    #viz_class = LeavenoutViz.LeavenoutViz(predictor, vocab, dataset_reader)
    viz_class = LeaveoneoutViz.LeaveoneoutViz(predictor, vocab, dataset_reader)
    #viz_class = SaliencymapViz.SaliencymapViz(predictor, vocab, dataset_reader)

    while True:
        sent = input("Enter a text: ")
        if sent == "exit" or sent == "quit":
            break

        output = viz_class.classify_sample(sent)
        print(output)
        tokens, weights, orig_label, orig_label_index = viz_class.get_heatmap_weights(sent)

        label = int(orig_label[0])
        viz_class.visualize_weights(tokens, weights, label, ["sadness", "joy", "love", "anger", "fear", "surprise"])