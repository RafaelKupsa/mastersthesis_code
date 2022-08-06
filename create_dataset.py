from src.dataset import DatasetForTLMandALP

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("target_language")
    parser.add_argument("source_languages", nargs="+")
    parser.add_argument("--eflomal_path", help="path to eflomal's align.py file")
    parser.add_argument("--max_len", default=256, help="maximum length for the tokenizer")
    parser.add_argument("--pre_encode", action="store_false")
    parser.add_argument("--many_to_many", action="store_false")
    parser.add_argument("-s", "--save_directory", default="data/datasets")
    parser.add_argument("--pbc_directory", default="data/pbc", help="directory storing the Parallel Bible Corpus (PBC)")
    parser.add_argument("--alignments_directory", default="data/alignments")
    args = parser.parse_args()

    print("Creating dataset for TLM and ALP")
    DatasetForTLMandALP(
        target_language=args.target_language,
        source_languages=args.source_languages,
        max_len=args.max_len,
        eflomal_path=args.eflomal_path,
        many_to_many=args.many_to_many,
        pre_encode=args.pre_encode,
        pbc_directory=args.pbc_directory,
        alignments_directory=args.alignments_directory,
        save_directory=args.save_directory
    )

