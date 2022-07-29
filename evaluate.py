import os.path
from collections import OrderedDict

import torch
from transformers import TrainingArguments, Trainer, set_seed
from transformers import DataCollatorForTokenClassification
from transformers import XLMRobertaForTokenClassification

from src.evaluation import DatasetForPOS, compute_metrics, POS_TAGS

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("eval_languages", nargs="+")
    parser.add_argument("--finetuning_language", default="eng")
    parser.add_argument("--max_len", default=256)
    parser.add_argument("-e", "--epochs", default=5)
    parser.add_argument("-b", "--batch_size", default=8)
    parser.add_argument("-l", "--learning_rate", default=2e-5)
    parser.add_argument("-g", "--gradient_accumulation_steps", default=4)
    parser.add_argument("-s", "--save_directory", default="eval_out")
    parser.add_argument("--num_seeds", default=5)
    parser.add_argument("--eval_steps", default=50)
    parser.add_argument("--logging_steps", default=25)
    parser.add_argument("--save_total_limit", default=3)
    args = parser.parse_args()

    # Dataset
    print("Loading datasets for POS-Tagging")
    train_dataset = DatasetForPOS(args.finetuning_language, "train", args.max_len)
    eval_dataset = DatasetForPOS(args.finetuning_language, "dev", args.max_len)
    test_datasets = {language: DatasetForPOS(language, "test", args.max_len) for language in args.eval_languages}

    # Data Collator
    print("Initializing data collator")
    collator = DataCollatorForTokenClassification(tokenizer=train_dataset.tokenizer, padding='longest')

    # Training Arguments
    print("Setting up training arguments")
    warmup_steps = int((args.epochs * (len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps))) * .01)

    training_arguments = TrainingArguments(
        output_dir=args.save_directory,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        metric_for_best_model="eval_accuracy",
        load_best_model_at_end=False,
        disable_tqdm=False
    )
    print("Device:", training_arguments.device)

    # Model
    print("Loading model")
    model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(POS_TAGS))
    if not os.path.exists(args.model_path) and args.model_path != "xlm-roberta-base":
        raise FileNotFoundError

    if os.path.exists(args.model_path):
        model_state_dict = torch.load(args.model_path, map_location=training_arguments.device)
        model_state_dict = OrderedDict([
            (key.replace('roberta.', '').replace('xlmr.', ''), value)
            for key, value in model_state_dict.items()
            if (key.startswith('roberta') or key.startswith('xlmr.')) and '.pooler.' not in key
        ])
        model.roberta.load_state_dict(model_state_dict)

    # Training
    print("Training start!")
    accuracies = {language: [] for language in args.eval_languages}
    f1_scores = {language: [] for language in args.eval_languages}
    for seed in map(lambda i: i * 10, range(args.num_seeds)):
        set_seed(seed)

        # Fine-tuning
        trainer = Trainer(
            model=model,
            data_collator=collator,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )
        trainer.train()

        # Evaluation
        for language in args.eval_languages:
            results = trainer.predict(test_datasets[language]).metrics
            accuracies[language].append(results["test_accuracy"])
            f1_scores[language].append(results["test_f1"])

    # Show Results
    print("Getting results")
    best_accuracies = {language: max(acc) for language, acc in accuracies.items()}
    avg_accuracies = {language: sum(acc)/len(acc) for language, acc in accuracies.items()}
    best_f1s = {language: max(f1) for language, f1 in f1_scores.items()}
    avg_f1s = {language: sum(f1) / len(f1) for language, f1 in f1_scores.items()}

    print("RESULTS".center(100, "-"))
    for language in args.eval_languages:
        print(language)
        print("\tbest_accuracy:", best_accuracies[language])
        print("\tavg_accuracy:", avg_accuracies[language])
        print("\tbest_f1:", best_f1s[language])
        print("\tavg_f1:", avg_f1s[language])
        print()





