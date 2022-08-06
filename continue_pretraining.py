import os.path

from transformers import TrainingArguments, Trainer

from src.collator import DataCollatorForTLMandALP, DataCollatorForTLM, DataCollatorForALP
from src.model import XLMRobertaForTLMandALP, XLMRobertaForALP, XLMRobertaForTLM
from src.config import XLMRobertaForALPConfig, XLMRobertaForTLMandALPConfig
from src.dataset import DatasetForTLMandALP

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--tlm", action="store_false")
    parser.add_argument("--alp", action="store_false")
    parser.add_argument("-e", "--epochs", default=80)
    parser.add_argument("-b", "--batch_size", default=4)
    parser.add_argument("-l", "--learning_rate", default=2e-5)
    parser.add_argument("-g", "--gradient_accumulation_steps", default=16)
    parser.add_argument("-s", "--save_directory", default="models")
    parser.add_argument("-n", "--save_name", default=None)
    parser.add_argument("-c", "--checkpoint", default=None)
    parser.add_argument("--save_steps", default=300)
    parser.add_argument("--logging_steps", default=50)
    parser.add_argument("--save_total_limit", default=3)
    args = parser.parse_args()

    # Dataset
    print("Loading dataset for TLM/ALP")
    dataset = DatasetForTLMandALP(from_file=args.dataset)

    # Data Collator
    print("Initializing data collator")
    if args.alp and args.tlm:
        collator = DataCollatorForTLMandALP(
            tokenizer=dataset.tokenizer,
            padding="longest"
        )
    elif args.alp:
        collator = DataCollatorForALP(
            tokenizer=dataset.tokenizer,
            padding="longest"
        )
    elif args.tlm:
        collator = DataCollatorForTLM(
            tokenizer=dataset.tokenizer
        )
    else:
        raise ValueError(f"Must either train on agx or tlm objective or both.")

    # Training Arguments
    tlm = "-tlm" if args.tlm else ""
    alp = "-alp" if args.alp else ""
    save_name = args.save_name if args.save_name else args.dataset.replace("dataset-tlm-alp", f"model{tlm}{alp}")
    savepath = os.path.join(args.save_directory, save_name)
    warmup_steps = int((args.epochs * (len(dataset) // (args.batch_size * args.gradient_accumulation_steps))) * .01)

    training_arguments = TrainingArguments(
        output_dir=savepath,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        prediction_loss_only=True,
        evaluation_strategy='no',
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        dataloader_num_workers=0,
        disable_tqdm=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    print("Device:", training_arguments.device)

    # Model
    if args.alp and args.tlm:
        if args.checkpoint:
            model = XLMRobertaForTLMandALP.from_pretrained(args.checkpoint)
        else:
            model = XLMRobertaForTLMandALP(XLMRobertaForTLMandALPConfig(vocab_size=dataset.tokenizer.vocab_size))
    elif args.alp:
        if args.checkpoint:
            model = XLMRobertaForALP.from_pretrained(args.checkpoint)
        else:
            model = XLMRobertaForALP(XLMRobertaForALPConfig())
    elif args.tlm:
        if args.checkpoint:
            model = XLMRobertaForTLM.from_pretrained(args.checkpoint)
        else:
            model = XLMRobertaForTLM.from_pretrained('xlm-roberta-base')
    else:
        raise ValueError(f"Must either train on agx or tlm objective or both.")

    # Training
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        data_collator=collator
    )

    if args.checkpoint:
        trainer.train(args.checkpoint)
    else:
        trainer.train()

    # Saving
    model.save_pretrained(savepath + "final_model/")