import random

import torch
from transformers.data.data_collator import DataCollatorMixin
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling

DataCollatorForALP = DataCollatorWithPadding


class DataCollatorForTLMandALP(DataCollatorMixin):
    """
    Data collator for simultaneous TLM (Translation Language Modeling) and ALP (Alignment Link Prediction)

    Batches have a 50-to-50 chance to be collated to TLM examples or ALP examples
    """
    def __init__(self, padding, tokenizer, mlm_probability=0.15):
        """
        Initializes the collator
        :param padding: padding strategy for ALP batches
        :param tokenizer: the tokenizer
        :param mlm_probability: masking probability
        """
        super().__init__()

        self.tlm_collator = DataCollatorForTLM(tokenizer=tokenizer, mlm_probability=mlm_probability)
        self.alp_collator = DataCollatorForALP(tokenizer=tokenizer, padding=padding)

    def __call__(self, examples):
        """
        Returns a collated version of the given examples
        :param examples: batch of examples
        :return: the collated batch
        """
        if random.random() < 0.5:
            batch = self.tlm_collator(examples)
            batch.pop("edge_index")
        else:
            batch = self.alp_collator(examples)

        return batch


class DataCollatorForTLM(DataCollatorForLanguageModeling):
    """
    Data collator for TLM (Translation Language Modeling) which works like Masked Language Modeling

    Adapted from Ebrahimi & Kann (2021) https://arxiv.org/abs/2106.02124
    """
    def __init__(self, tokenizer, mlm_probability=0.15):
        """
        Initializes the collator
        :param tokenizer: the tokenizer
        :param mlm_probability: masking probability
        """
        super().__init__(mlm=True, mlm_probability=mlm_probability, tokenizer=tokenizer)

    def __call__(self, examples):
        """
        Returns a collated version of the given examples
        :param examples: batch of examples
        :return: the collated batch
        """
        self.decisions = []
        for example in examples:
            example_length = int(example['input_ids'].shape[0])
            self.decisions.append([True] * (example_length // 2) + [False] * (example_length - example_length // 2))

        batch = self.tokenizer.pad(examples, return_tensors="pt")
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        batch["input_ids"], batch["labels"] = self.mask_tokens(batch["input_ids"], special_tokens_mask=special_tokens_mask)

        return batch

    def mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Creates masked inputs and unmasked labels from inputs with the given special_tokens_mask
        80% masked, 10% random, 10% original.
        :param inputs: the input batch
        :param special_tokens_mask: mask token
        :return: masked inputs batch and unmasked labels batch
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        x = int(labels.shape[1])
        self.tensor_decisions = []
        for dec in self.decisions:
            self.tensor_decisions.append(torch.BoolTensor(dec + [True] * (x - len(dec))).unsqueeze(0))

        decision_tensor = torch.cat(self.tensor_decisions, dim=0)

        probability_matrix[decision_tensor == True] = torch.FloatTensor([0])

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels
