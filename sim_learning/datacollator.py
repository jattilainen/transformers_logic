import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

REPLACE_NONE = -100


@dataclass
class DataCollatorForSim:
    """
    Data collator used for simultaneous learning.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for both sentence classification and masked language modeling
    """
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    block_size: int = 512
    entropy: bool = False 

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)
        block_size = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=False)

        ori_sent = []
        ori_mask = []
        class_label = []

        for example in examples:
            ori_sent += [torch.tensor(example["input_ids"][:block_size], dtype=torch.long)]
            ori_mask += [torch.ones(len(example["input_ids"][:block_size]))]

            class_label += [torch.tensor([example["label"]], dtype=torch.long)]

        input_ids = ori_sent + ori_sent
        attention_mask = ori_mask + ori_mask

        assert len(input_ids) == batch_size * 2
        assert len(attention_mask) == batch_size * 2

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        if not self.entropy:
            mlm_sent, mlm_label = self.mask_tokens(input_ids[batch_size:])
            input_ids[batch_size:] = mlm_sent
        else:
            shuffle_sent, mlm_label = self.mask_tokens(input_ids[batch_size:])
            # mlm_label is zero buffer with no sense
            input_ids[batch_size:] = shuffle_sent
        labels = torch.cat((torch.tensor(class_label).view(-1, 1), mlm_label), dim=1)
        a, b = input_ids[:batch_size], input_ids[batch_size:]
        input_ids = torch.cat((a, b), dim=1)
        a, b = attention_mask[:batch_size], attention_mask[batch_size:]
        attention_mask = torch.cat((a, b), dim=1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = REPLACE_NONE  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def shuffle_tokens(self, inputs: torch.Tensor, ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare shuffled tokens inputs/labels: 50% shuffle.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = torch.zeros(inputs.shape, dtype=torch.long)
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        shuffle_mask = torch.ones(labels.shape)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
        ]
        shuffle_mask.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            shuffle_mask.masked_fill_(padding_mask, value=0.0)

        for i in range(len(inputs)):
            sentence_indices = torch.nonzero(shuffle_mask[i])
            r = torch.randperm(len(sentence_indices))
            inputs[i][sentence_indices] = inputs[i][sentence_indices[r]]

        return inputs, labels
