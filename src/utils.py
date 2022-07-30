import os
import pandas as pd

POS_TAGS = [
    'ADJ',
    'ADP',
    'PUNCT',
    'ADV',
    'AUX',
    'SYM',
    'INTJ',
    'CCONJ',
    'X',
    'NOUN',
    'DET',
    'PROPN',
    'NUM',
    'VERB',
    'PART',
    'PRON',
    'SCONJ',
    '_'
]


def lang2pos(lang, train_dev_test):
    return "data/pos/" + {
        'eng': 'UD_English-GUM/en_gum-ud-{}.conllu',
        'fra': 'UD_French-GSD/fr_gum-ud-{}.conllu',
        'arb': 'UD_Arabic-NYUAD/ar_nyuad-ud-{}.conllu',
        'ell': 'UD_Greek-GDT/el_gdt-ud-{}.conllu',
        'hun': 'UD_Hungarian-Szeged/hu_szeged-ud-{}.conllu',
        'gle': 'UD_Irish-IDT/ga_idt-ud-{}.conllu',
        'bam': 'UD_Bambara-CRB/bm_crb-ud-{}.conllu',
        'cop': 'UD_Coptic-Scriptorium/cop_scriptorium-ud-{}.conllu',
        'glv': 'UD_Manx-Cadhan/gv_cadhan-ud-{}.conllu',
        'grc': 'UD_Ancient_Greek-PROIEL/grc_proiel-ud-{}.conllu',
        'mlt': 'UD_Maltese-MUDT/mt_mudt-ud-{}.conllu',
        'myv': 'UD_Erzya-JR/myv_jr-ud-{}.conllu',
        'wol': 'UD_Wolof-WTB/wo_wtb-ud-{}.conllu',
        'yor': 'UD_Yoruba-YTB/yo_ytb-ud-{}.conllu'
    }[lang].format(train_dev_test)


def lang2bible(lang):
    """
    get the path to a bible from an iso-639-3 language code
    :param lang: the language (iso-639-3 code)
    :return: the path to the corresponding bible
    """
    pbc_directory = "data/pbc"

    for file in os.listdir(pbc_directory):
        if file[:3] == lang:
            return os.path.join(pbc_directory, file)

    raise FileNotFoundError


def languages_by_relation_to(source_language):
    """
    Returns a list of related languages to the given source language, sorted by degree of relation

    Relation is calculated as the length of the common family tree
    :param source_language: the source language (iso-639-3 code)
    :return: list of related languages in order (iso-639 codes)
    """
    lang_info = pd.read_csv("data/languages_info.csv")
    source_lang_tree = {lang["iso639-3"]: lang["family-tree"] for lang in lang_info.to_dict("index").values()}[source_language]

    def common_initial_substring_length(tree: str):
        if type(tree) == float:
            return -1
        for i in range(len(tree), -1, -1):
            if source_lang_tree.startswith(tree[:i]):
                return tree[:i].rfind(",")

    lang_info["similarity"] = lang_info["family-tree"].map(common_initial_substring_length)
    related_languages = lang_info[lang_info["similarity"] > 0]
    related_languages = related_languages.sort_values("similarity", ascending=False, ignore_index=True)

    return related_languages["iso639-3"].tolist()
