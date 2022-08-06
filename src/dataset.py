import os.path
import itertools
from collections import defaultdict
import random

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer

from .alignment import MultiAligner
from .utils import lang2bible


class DatasetForTLMandALP(Dataset):
    """
    Dataset class for TLM (Translation Language Modeling) and ALP (Alignment Link Prediction)
    """
    def __init__(
            self,
            from_file=None,
            target_language=None,
            source_languages=None,
            max_len=None,
            eflomal_path=None,
            many_to_many=False,
            pre_encode=True,
            pbc_directory="data/pbc",
            alignments_directory="data/alignments",
            save_directory="data/datasets"
    ):
        """
        Loads a given dataset (from_file) or creates a new one with the given specifications
        :param from_file: path of file containing a valid dataset
        :param target_language: the target language (iso-639-3 code)
        :param source_languages: list of source languages (iso-639-3 code)
        :param max_len: maximum length of the tokenizer (XLMRobertaTokenizer)
        :param eflomal_path: path to the eflomal alignment python script (align.py)
        :param many_to_many: whether to create many-to-many training examples (also between target languages)
        :param pre_encode: whether to pre-encode the dataset
        :param pbc_directory: path to the Parallel Bible Corpus (PBC) directory
        :param alignments_directory: path to the alignments directory; alignments are created if not present
        :param save_directory: path to the directory to save the dataset
        """
        super().__init__()

        self.target_language = target_language
        self.source_languages = source_languages
        self.max_len = max_len

        self.many_to_many = many_to_many
        self.pre_encode = pre_encode

        self.max_nodes = 512
        self.max_edges = 200

        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', max_len=max_len)

        self.eflomal_path = eflomal_path
        self.pbc_directory = pbc_directory
        self.alignments_directory = alignments_directory
        self.save_directory = save_directory

        self.verses = {}
        self.alignments = {}
        self.examples = []

        # LOAD EXISTING DATASET if possible
        if from_file:
            if os.path.exists(from_file):
                dataset = torch.load(from_file)
                self.pre_encode = dataset["pre-encoded"]
                self.examples = dataset["examples"]
                return
            raise ValueError(f"{from_file} is not a valid dataset.")

        # CREATE NEW DATASET otherwise
        all_languages = list(sorted([self.target_language] + self.source_languages))

        if not os.path.exists(self.alignments_directory):
            os.mkdir(self.alignments_directory)

        aligner = MultiAligner(eflomal_path, max_len=max_len)
        aligner.align(all_languages, self.alignments_directory)

        # Loading verses
        with tqdm(total=len(all_languages)) as progressbar:
            for language in all_languages:
                progressbar.set_description(f"Loading verses for {language}")
                self._load_verses(language)
                progressbar.update()

        # Loading alignments
        if self.many_to_many:
            lang_combinations = list(itertools.combinations([self.target_language] + self.source_languages, 2))
        else:
            lang_combinations = [(self.target_language, source_language) for source_language in self.source_languages]

        with tqdm(total=len(lang_combinations)) as progressbar:
            for lang1, lang2 in lang_combinations:
                progressbar.set_description(f"Loading alignments for {lang1}-{lang2}")
                self._load_alignments(lang1, lang2)
                progressbar.update()

        # Creating training examples
        with tqdm(total=len(lang_combinations)) as progressbar:
            for lang1, lang2 in lang_combinations:
                progressbar.set_description(f"Creating training examples for {lang1}-{lang2}")
                self._create_examples(lang1, lang2)
                progressbar.update()

        random.seed(50)
        random.shuffle(self.examples)

        lgs = f"{target_language}--{'-'.join(source_languages)}"
        ml = f"_ml{max_len}"
        mtm = "_many-to-many" if many_to_many else ""
        savepath = os.path.join(save_directory, f"dataset-tlm-alp_{lgs}{ml}{mtm}.pt")

        torch.save({
            "pre-encoded": self.pre_encode,
            "examples": self.examples
        }, savepath)
        print(f"Saved to {savepath}.")

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
        if self.pre_encode:
            return self.examples[idx]
        else:
            return self._encode(self.examples[idx])

    def _load_verses(self, language):
        """
        Loads the verses into the dataset for a given language
        :param language: the language (iso-639-3 code)
        """
        bible = lang2bible(language)

        self.verses[language] = {}
        with open(bible) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                split_line = line.strip().split("\t")
                if len(split_line) != 2:
                    continue
                verse_id = split_line[0]
                if int(verse_id) < 40001001:
                    continue
                verse = split_line[1]
                self.verses[language][verse_id] = verse

    def _load_alignments(self, lang1, lang2):
        """
        Loads the alignments into the dataset for a given language pair
        :param lang1: language 1 (iso-639-3 code)
        :param lang2: language 2 (iso-639-3 code)
        """
        lang1, lang2 = tuple(sorted([lang1, lang2]))
        l1l2 = f"{lang1}-{lang2}"
        self.alignments[l1l2] = defaultdict(dict)

        verse_idx_path = os.path.join(self.alignments_directory, f"{l1l2}_verses")
        forward_links_path = os.path.join(self.alignments_directory, f"{l1l2}_forward-links")
        reverse_links_path = os.path.join(self.alignments_directory, f"{l1l2}_reverse-links")

        with open(verse_idx_path) as f:
            verse_idx = [line.strip() for line in f]

        for direction, path in [("forward", forward_links_path), ("reverse", reverse_links_path)]:
            with open(path) as f:
                for i, line in enumerate(f):
                    if int(verse_idx[i]) < 40001001:
                        continue
                    alignments = [tuple(alignment.strip().split('-')) for alignment in line.strip().split()]
                    self.alignments[l1l2][verse_idx[i]][direction] = alignments

    def _create_examples(self, target_language, source_language):
        """
        Creates training examples for a target language and a source language
        :param target_language: the target language (iso-639-3 code)
        :param source_language: the source language (iso-639-3 code)
        """
        lang1, lang2 = tuple(sorted([target_language, source_language]))
        l1l2 = f"{lang1}-{lang2}"

        for i, verse_id in enumerate(self.verses[lang1]):
            if verse_id not in self.verses[lang2]:
                continue
            if verse_id not in self.verses[self.target_language]:
                continue
            if verse_id not in self.alignments[l1l2]:
                continue

            tokens = (self.verses[source_language][verse_id], self.verses[target_language][verse_id])
            tokens_encoded = (self.tokenizer(tokens[0], truncation=True, max_length=self.max_nodes, return_tensors='pt'),
                              self.tokenizer(tokens[1], truncation=True, max_length=self.max_nodes, return_tensors='pt'))
            len_source = min([tokens_encoded[0]['input_ids'].size(1) - 2, self.max_nodes // 2 - 1])
            len_target = min([tokens_encoded[1]['input_ids'].size(1) - 2, self.max_nodes // 2 - 1])

            positive_edges = [[], []]
            for t, s in self.alignments[l1l2][verse_id]["forward"]:
                t, s = int(t) + 1, int(s) + 1  # + 1 to account for <s>
                if lang1 == target_language:
                    t += len_source
                    if s < self.max_nodes / 2 and t < self.max_nodes - 1:
                        positive_edges[0].append(t)
                        positive_edges[1].append(s)
                else:
                    s += len_source
                    if t < self.max_nodes / 2 and s < self.max_nodes - 1:
                        positive_edges[0].append(t)
                        positive_edges[1].append(s)
            for t, s in self.alignments[l1l2][verse_id]["reverse"]:
                t, s = int(t) + 1, int(s) + 1  # + 1 to account for <s>
                if lang1 == target_language:
                    t += len_source
                    if s < self.max_nodes / 2 and t < self.max_nodes - 1:
                        positive_edges[0].append(s)
                        positive_edges[1].append(t)
                else:
                    s += len_source
                    if t < self.max_nodes / 2 and s < self.max_nodes - 1:
                        positive_edges[0].append(s)
                        positive_edges[1].append(t)

            if not positive_edges[0]:
                continue

            example = {"tokens": tokens, "pos_edges": positive_edges, "len_target": len_target, "len_source": len_source}
            if self.pre_encode:
                example = self._encode(example)

            self.examples.append(example)

    def _encode(self, example):
        """
        Encodes an example
        :param example: the example dictionary
        :return: a dictionary with input_ids, attention_mask, edge_index and labels
        """
        enc = self.tokenizer(
            example["tokens"][0] + " " + example["tokens"][1],
            truncation=True,
            max_length=self.max_nodes,
            return_tensors='pt'
        )

        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)

        pos_edges = torch.LongTensor(self._pad_positive_edges(example['pos_edges']))
        neg_edges = torch.LongTensor(self._sample_negative_edges(example))
        edges = torch.concat([pos_edges, neg_edges], dim=-1)
        edges_labels = torch.tensor([1] * len(pos_edges[0]) + [0] * len(neg_edges[0]), dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "edge_index": edges,
            "labels": edges_labels
        }

    def _pad_positive_edges(self, edges):
        """
        Pads (with duplicated edges) or truncates the given edges to half the maximum number of edges
        :param edges: a list containing two lists: the edge sources and the edge targets
        :return: padded or truncated list
        """
        padded_edge_idcs = []
        while len(padded_edge_idcs) < int(self.max_edges / 2):
            padded_edge_idcs = np.append(padded_edge_idcs, np.random.permutation(np.arange(len(edges[0]))))
        padded_edge_idcs = padded_edge_idcs[:int(self.max_edges / 2)].astype(np.int64)

        return [[edges[0][idx] for idx in padded_edge_idcs], [edges[1][idx] for idx in padded_edge_idcs]]

    def _sample_negative_edges(self, example):
        """
        Creates negative alignment links for a given example (half of the maximum number of edges)
        :param example: the example dictionary
        :return: a list containing two lists: the edge sources and the edge targets
        """
        pos_edges = [zip(example['pos_edges'][0], example['pos_edges'][1])]
        neg_edges = [[], []]
        while len(neg_edges[0]) < int(self.max_edges / 2):
            x = np.random.choice(example["len_source"]) + 1
            y = np.random.choice(example["len_target"]) + example["len_source"] + 1
            start = x if np.random.random() <= 0.5 else y
            end = x if start == y else y
            if (start, end) not in pos_edges:
                neg_edges[0].append(start)
                neg_edges[1].append(end)

        return neg_edges
