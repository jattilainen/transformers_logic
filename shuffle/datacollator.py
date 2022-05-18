import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

REPLACE_NONE = -100


@dataclass
class DataCollatorForShuffle:
    """
    Data collator used for linguistic error correction task.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for both masked language modeling and linguistic error correction
    """
    tokenizer: PreTrainedTokenizerBase
    shuffle_probability: float = 0.5
    block_size: int = 128
    shuffle: bool = True
    predict_tokens: bool = True

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)
        block_size = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=False)

        sent = []
        mask = []
        label = []

        for example in examples:
            sent += [torch.tensor(example["input_ids"][:block_size], dtype=torch.long)]
            mask += [torch.ones(len(example["input_ids"][:block_size]))]

        assert len(sent) == batch_size
        assert len(mask) == batch_size

        input_ids = pad_sequence(sent, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(mask, batch_first=True, padding_value=0)
        
        if self.shuffle:
            shuffle_sent, shuffle_label = self.shuffle_tokens(input_ids)

        return {
            "input_ids": shuffle_sent,
            "attention_mask": attention_mask,
            "labels": shuffle_label
        }

    def shuffle_tokens(self, inputs: torch.Tensor, ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare shuffled tokens inputs/labels: 50% shuffle.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        if self.predict_tokens:
            labels = inputs.clone()
        else:
            labels = torch.ones(inputs.shape, dtype=torch.long)
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.shuffle_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        shuffle_indices = torch.bernoulli(probability_matrix).bool()
        if self.predict_tokens:
            labels[~shuffle_indices] = REPLACE_NONE
        else:
            labels[~shuffle_indices] = 0
        for i in range(len(inputs)):
            sentence_indices = torch.nonzero(shuffle_indices[i])
            r = torch.randperm(len(sentence_indices))
            inputs[i][sentence_indices] = inputs[i][sentence_indices[r]]

        return inputs, labels
