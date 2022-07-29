import conllu
from seqeval.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer

from .utils import lang2pos, POS_TAGS


def compute_metrics(model_output):
    """
    Computes POS_Tagging metrics (F1 score, Accuracy) for a given model output
    :param model_output: the model output
    :return: a dictionary with values for accuracy and f1
    """
    all_labels = model_output.label_ids
    all_predictions = model_output.predictions.argmax(-1)

    corrected_predictions, corrected_labels = [], []

    for labels, predictions in zip(all_labels, all_predictions):
        discard = [label == -100 for label in labels]
        corrected_predictions.append([
            POS_TAGS.index(prediction)
            for discard_label, prediction
            in zip(discard, predictions)
            if not discard_label
        ])
        corrected_labels.append([
            POS_TAGS.index(label)
            for discard_label, label
            in zip(discard, labels)
            if not discard_label
        ])

    return {
        'accuracy': accuracy_score(corrected_labels, corrected_predictions) * 100,
        'f1': f1_score(corrected_labels, corrected_predictions) * 100
    }


class DatasetForPOS(Dataset):
    """
    Dataset class for POS-Tagging of Universal Dependencies corpus
    """
    def __init__(self, language, train_dev_test, max_len):
        """
        Loads the dataset
        :param language: the language (iso-639-3 code)
        :param train_dev_test: whether to load the train, the dev or the test set
        :param max_len: maximum length of the tokenizer (XLMRobertaTokenizer)
        """
        self.language = language
        self.max_len = max_len
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', max_len=max_len)
        self.label2id = {tag: i for i, tag in enumerate(POS_TAGS)}

        self.examples = self.read_file(lang2pos(language, train_dev_test))
        self.label_masks = [[] for _ in range(len(self.examples))]

    def __len__(self):
        """
        Returns the length of the dataset (number of examples)
        :return: integer
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns an encoded example for training
        :param idx: integer index
        :return: encoded example dictionary
        """
        return self.encode(self.examples[idx])

    def read_file(self, file):
        """
        Parses the dataset from the file
        :param file: the file (in .conllu format)
        :return: a list of training examples
        """
        examples = []

        with open(file) as f:
            for tokens in conllu.parse_incr(f):
                try:
                    forms, labels = zip(*[(token['form'], self.label2id[token['upos']]) for token in tokens])
                    examples.append((forms, labels))
                except KeyError:
                    continue

        return examples

    def encode(self, example):
        """
        Encodes an example
        :param example: the example dictionary
        :return: a dictionary with input_ids, attention_mask and labels
        """
        expanded_labels = []

        for form, label in zip(*example):
            subwords = self.tokenizer.tokenize(form)
            expanded_labels.extend([-100] * (len(subwords) - 1) + [label])

        encoded = self.tokenizer(
            ' '.join(example[0]),
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=True,
        )

        encoded['labels'] = expanded_labels[:self.max_len]

        return encoded
