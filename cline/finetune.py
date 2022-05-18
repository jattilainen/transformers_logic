import os
import sys
print(sys.prefix)
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback
)


from transformers import (
    HfArgumentParser,
    PreTrainedTokenizer
)

def preprocess_snli(examples):
    x = tokenizer(examples["hypothesis"], examples["premise"])
    return x

import numpy as np
from datasets import load_metric
import numpy as np
import torch.nn as nn
import torch

def fix_negative(examples):
    narr = np.array(examples['label'])
    narr[narr < 0] *= -1
    examples['label'] = narr
    return examples

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

@dataclass
class PreprocessArguments:
    model_name: Optional[str] = field(
        default='roberta-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    model_path: Optional[str] = field(
        default='roberta-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    task_name: Optional[str] = field(
        default='snli', metadata={"help": "Task to finetune - snli or mnli"}
    )
    cache_dir: Optional[str] = field(
        default='/home/vapavlov_4/.cache', metadata={"help": "Cache dir"}
    )
    output_dir: Optional[str] = field(
        default='exps/exp0', metadata={"help": "Where do you want to store results"}
    )
    batch_size: Optional[int] = field(
        default=32, metadata={"help": "Per device batch size"}
    )
    eval_batch_size: Optional[int] = field(
        default=128, metadata={"help": "Per device batch size"}
    )
    exp_name: Optional[str] = field(
        default='exp0', metadata={"help": "Experiment name for wandb log"}
    )

class TrainMetricsCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

import torch
        
if __name__ == "__main__":
    n_gpus = max(1, torch.cuda.device_count())
    print("Found {} GPUs".format(n_gpus))
    parser = HfArgumentParser((PreprocessArguments,))
    model_args, = parser.parse_args_into_dataclasses()
    name = model_args.model_name
    
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=model_args.cache_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_path, num_labels=3, cache_dir=model_args.cache_dir)
    metric = load_metric("accuracy")
    set_seed(42)

    if model_args.task_name == 'snli':
        snli = load_dataset("snli", cache_dir=model_args.cache_dir)
        tokenized_snli = snli.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=["premise", "hypothesis"])
        train_dataset = tokenized_snli["train"]
        eval_dataset = tokenized_snli["validation"]
        test_datasets = [("test", tokenized_snli["test"])]
    if model_args.task_name == 'mnli':
        mnli = load_dataset("multi_nli", cache_dir=model_args.cache_dir)
        # not typo in preprocess_snli, because they both have hypothesis and premise
        tokenized_mnli = mnli.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'])
        dataset = tokenized_mnli["train"].shuffle(42)
        train_valid = dataset.train_test_split(test_size=0.1)
        train_dataset = train_valid['train']
        eval_dataset = train_valid['test']
        test_datasets = [("validation_matched", tokenized_mnli["validation_matched"]), ("validation_mismatched", tokenized_mnli["validation_mismatched"])]
        
    training_args = TrainingArguments(
        output_dir=os.path.join(model_args.output_dir, '{}_finetune_logs'.format(model_args.task_name)),
        per_device_train_batch_size=model_args.batch_size // n_gpus,
        per_device_eval_batch_size=model_args.eval_batch_size // n_gpus,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy='epoch',
        num_train_epochs=7,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        seed=42,
        learning_rate=2e-5,
        weight_decay=0.01,  # strength of weight decay
        max_grad_norm=1.0,
        report_to="wandb",
        run_name=model_args.exp_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(TrainMetricsCallback(trainer)) 
    trainer.train()
    
    with open(os.path.join(model_args.output_dir, '{}_results.txt'.format(model_args.task_name)), 'w') as f:
        for test_name, test_dataset in test_datasets:
            res = trainer.predict(test_dataset)
            if trainer.is_world_process_zero():
                print(test_name, res.metrics['test_accuracy'], file=f)
                trainer.save_model(os.path.join(model_args.output_dir, '{}_{}_final_model'.format(model_args.task_name, test_name)))
