import os
import itertools

from tqdm import tqdm
from transformers import XLMRobertaTokenizer

from .utils import lang2bible


class Aligner:
    """
    Class for aligning two bibles from the Parallel Bible Corpus with eflomal on the subword level

    Use $Aligner.align(lang1, lang2, savepath) to align two languages
    which creates a file with parallel sentences in the bible, a verse index file,
    forward links, reverse links, forward scores and reverse scores
    """
    def __init__(self, eflomal_path, max_len):
        """
        Constructor for the Aligner
        :param eflomal_path: path to the eflomal alignment python script (align.py)
        :param max_len: maximum length of the tokenizer (XLMRobertaTokenizer)
        """
        self.eflomal_path = eflomal_path
        self.max_len = max_len

        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', max_len=max_len)

    def align(self, lang1, lang2, savepath):
        """
        Creates subword alignments for the bibles of two languages

        :param lang1: language 1 (iso639-3 code)
        :param lang2: language 2 (iso639-3 code)
        :param savepath: path where to store the files
        """
        if os.path.exists(savepath + "_forward-links") and os.path.exists(savepath + "_reverse-links"):
            return

        self.prepare_files(lang1, lang2, savepath)
        os.system(f"python3 {self.eflomal_path} "
                  f"-i {savepath}_parallel "
                  f"-f {savepath}_forward-links "
                  f"-r {savepath}_reverse-links "
                  f"-F {savepath}_forward-scores "
                  f"-R {savepath}_reverse-scores")

    def prepare_files(self, lang1, lang2, savepath):
        """
        Prepares the bible files for alignment with eflomal
        :param lang1: language 1 (iso639-3 code)
        :param lang2: language 2 (iso639-3 code)
        :param savepath: path where to store the files
        """
        verses_file = open(savepath + "_verses", 'w')

        with open(lang2bible(lang1), 'r') as f:
            verses_source = [
                tuple(line.strip().split("\t"))
                for line in f.readlines()
                if not line.startswith("#") and len(line.strip().split("\t")) == 2
            ]

        with open(lang2bible(lang2), 'r') as f:
            verses_target = {
                line.strip().split("\t")[0]: line.strip().split("\t")[1]
                for line in f.readlines()
                if not line.startswith("#") and len(line.strip().split("\t")) == 2
            }

        with open(savepath + "_parallel", 'w') as f:
            for verse_id, verse_source in verses_source:
                if verse_id not in verses_target:
                    continue

                verse_target = verses_target[verse_id]
                if len(verse_source) != 0 and len(verse_target) != 0:
                    f.write(" ".join(self.tokenizer.tokenize(verse_source.strip())))
                    f.write(" ||| ")
                    f.write(" ".join(self.tokenizer.tokenize(verse_target.strip())))
                    f.write("\n")
                    verses_file.write(str(verse_id) + "\n")

        verses_file.close()


class MultiAligner:
    """
    Class for aligning multiple bibles, in every binary combination, from the Parallel Bible Corpus with eflomal on the subword level

    Use $Aligner.align(languages, save_directory) to align the languages
    which creates a file with parallel sentences in the bible, a verse index file,
    forward links, reverse links, forward scores and reverse scores
    for every language pair
    """
    def __init__(self, eflomal_path, max_len):
        """
        Constructor for the Aligner
        :param eflomal_path: path to the eflomal alignment python script (align.py)
        :param max_len: maximum length of the tokenizer (XLMRobertaTokenizer)
        """
        self.eflomal_path = eflomal_path
        self.max_len = max_len

    def align(self, languages, save_directory):
        """
        Creates subword alignments for the bibles of multiple languages

        :param languages: list of languages to align (iso639-3 code)
        :param save_directory: directory where to store the files
        """
        lang_combinations = list(itertools.combinations(languages, 2))
        aligner = Aligner(self.eflomal_path, self.max_len)

        with tqdm(total=len(lang_combinations)) as progressbar:
            for lang1, lang2 in lang_combinations:
                progressbar.set_description(f"Creating alignments for {lang1}-{lang2}")
                aligner.align(lang1, lang2, os.path.join(save_directory, f"{lang1}-{lang2}"))
                progressbar.update()
