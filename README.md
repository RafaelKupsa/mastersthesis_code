# Improving Pretrained Multilingual Models for Low Resource Languages by Learning from Alignment Graphs
### Master's Thesis by Rafael Kupsa
### Center for Information and Language Processing, Ludwig-Maximilians-Universität München
### Supervised by Ayyoob Imani, M. Sc.

This repository contains the code used to conduct the experiments for the Master's Thesis in Computational Linguistics at LMU Munich, by Rafael Kupsa.
The goal of the thesis was to improve the Pretrained Multilingual Model [XLM-R](https://github.com/facebookresearch/XLM) for low-resource languages with a novel training objective called Alignment Link Prediction (ALP) and by investigating the role of the source language when using parallel data.

## Abstract of the paper

Pretrained language models are deep neural networks trained on large amounts of natural language texts. Their contextual representations can be adapted using transfer learning to perform a diverse range of tasks. Their multilingual versions -- trained on data in more than one language -- make cross-lingual transfer between those languages possible. Interestingly, they also tend to perform quite well for unseen languages which can be further incentivized by continued pretraining with unlabeled data. Some languages, however, are so poor in resources that more sophisticated methods are necessary. This thesis is an addition to ongoing research which aims to explore these methods in detail. I present a novel continued-pretraining approach that leverages automatically induced word alignments between parallel sentences (in a source language and a target language) and observe that it negatively influences the Part-of-Speech Tagging performance in the target language. I also investigate the benefits of including more than one source language which seems to have no significant effect. Experiments on the influence of different source languages show that continued pretraining with parallel sentences works best if the languages are related. There seems to be no significant benefit if the languages are represented in the original pretraining corpus of the base model. Finally, I show that the largest increases in performance can usually be achieved by altering the fine-tuning language.

## Usage

The experiments can be recreated and further explored with the code in this repository.

### Prerequisites

In a first step, clone this repository and install the requirements:

```bash
git clone https://github.com/RafaelKupsa/mastersthesis_code.git
cd mastersthesis_code
pip install -r requirements.txt
```

You will also need to install ['eflomal'](https://github.com/robertostling/eflomal) for creating alignments, see the repository README for instructions.

You will need access to the ['Parallel Bible Corpus' (PBC) by Mayer & Cysouw (2014)](https://aclanthology.org/L14-1215/) for performing continued pretraining. The files should be placed in the ['data/pbc'](data/pbc) folder.

For evaluation, download the ['Universal Dependencies'](https://universaldependencies.org/) datasets for your preferred target and fine-tuning languages and place them in the ['data/pos'](data/pos) folder.

### Creating datasets

With access to the PBC and eflomal, datasets can be created with the [`create_dataset.py`](create_dataset.py) script.

```bash
python3 create_dataset.py mlt eng ind rus --eflomal_path path_to_eflomal_align_script
```

Arguments:
* `target_language`: The target language's ISO 639-3 code
* `source_languages`: All following language codes after the first are treated as source languages, at least one is required
* `--eflomal_path`: The location of the `align.py` file in the eflomal repository you downloaded

Optional arguments:
* `--max_len`: The maximum length for the tokenizer, default is 256
* `--pre_encode`: Whether to pre-encode the training examples (saves time during training but negative edges stay the same each epoch), true by default
* `--many_to_many`: Whether to also create training examples for every source-language pair, true by default
* `-s` or `--save_directory`: Where to store the dataset, default is ['data/datasets'](data/datasets)
* `--pbc_directory`: Directory to the PBC dataset files, default is ['data/pbc'](data/pbc)
* `--alignments_directory`: Where to store the alignments created by eflomal, default is ['data/alignments'](data/alignments)

### Continued Pretraining

After creating a dataset, it can be used to perform continued pretraining on XLM-R with the [`continue_pretraining.py`](continue_pretraining.py) script.

```bash
python3 continue_pretraining.py dataset-tlm-alp_???--???-???-???_ml???_many-to-many.pt --tlm --alp
```

Arguments:
* `dataset`: The path to the dataset created with `create_dataset.py`

Optional arguments.
* `--tlm`: Include Translation Language Modeling (TLM) as a continued-pretraining objective, false by default (at least one of TLM or ALP is required)
* `--alp`: Include Alignment Link Prediction (ALP) as a continued-pretraining objective, false by default (at least one of TLM or ALP is required)
* `-e` or `--epochs`: Number of training epochs, default is 80
* `-b` or `--batch_size`: Batch size, default is 4
* `-l` or `--learning_rate`: Learning rate, default is 2e-5
* `-g` or `--gradient_accumulation_steps`: Gradient accumulation steps, default is 16
* `-s` or `--save_directory`: Where to store the checkpoints and the final model, default is ['models'](models)
* `-n` or `--save_name`: Name of the model, default is 'model{-tlm}{-alp}_TARGET--SOURCES_ml{256}{_many-to-many}'
* `-c` or `--checkpoint`: Path to a checkpoint to keep training from, default is None
* `--save_steps`: Number of steps between creating checkpoints, default is 300
* `--logging_steps`: Number of steps between displaying training loss, default is 50
* `--save_total_limit`: Number of checkpoints to keep at any given time (old checkpoints are deleted), default is 3

### Evaluation

After training a model, it can be evaluated on Part-of-Speech Tagging with the [`evaluate.py`](evaluate.py) script, provided access to the Universal Dependencies dataset in the target languages and a fine-tuning language

```bash
python3 evalutate.py model-tlm-alp_???--???-???-???_ml???_many-to-many/final_model/pytorch_model.bin mlt
```

Arguments:
* `model_path`: Path to the model (after training, it can be found in 'save_name/final_model/pytorch_model.bin' (you can also provide 'xlm-roberta-base' to evaluate the original model)
* `eval_languages`: ISO 639-3 language codes for the languages to evaluating, at least one is required

Optional arguments:
* `--finetuning_language`: Language for fine-tuning on the tagging task, default is English
* `--max_len`: The maximum length for the tokenizer, default is 256
* `-e` or `--epochs`: Number of fine-tuning epochs, default is 5
* `-b` or `--batch_size`: Batch size, default is 8
* `-l` or `--learning_rate`: Learning rate, default is 2e-5
* `-g` or `--gradient_accumulation_steps`: Gradient accumulation steps, default is 4
* `-s` or `--save_directory`: Where to store the final model, default is ['eval_out'](eval_out)
* `--num_seeds`: Number of seeds, final results provide an average over all runs, default is 5
* `--eval_steps`: Number of steps between evaluation during training process, default is 50
* `--logging_stpes`: Number of steps between displaying training loss, default is 25
* `--save_total_limit`: Number of checkpoint to keep at any given time (old checkpoints are deleted), default is 3
