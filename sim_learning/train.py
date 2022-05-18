# this file should be used to train cline on existing dataset
import random
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from datasets import Dataset
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer

from transformers import (
    HfArgumentParser,
    PreTrainedTokenizer
)

from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback
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
    predictions = np.argmax(eval_pred[0], axis=-1)
    return metric.compute(predictions=predictions, references=eval_pred[1][:, 0])

class TrainMetricsCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

@dataclass
class PreprocessArguments:
    model_name: Optional[str] = field(
        default='roberta-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='/home/vapavlov_4/.cache', metadata={"help": "Cache dir"}
    )
    output_dir: Optional[str] = field(
        default='/home/vapavlov_4/models/robeta_base_cline_snli', metadata={"help": "Where do you want to store model"}
    )
    mlm_layer2: Optional[bool] = field(
        default=False, metadata={"help": "Use mlm loss on layer 2"}
    )
    mlm_layer4: Optional[bool] = field(
        default=False, metadata={"help": "Use mlm loss on layer 4"}
    )
    mlm_layer6: Optional[bool] = field(
        default=False, metadata={"help": "Use mlm loss on layer 6"}
    )
    mlm_layer8: Optional[bool] = field(
        default=False, metadata={"help": "Use mlm loss on layer 8"}
    )
    mlm_layer10: Optional[bool] = field(
        default=False, metadata={"help": "Use mlm loss on layer 10"}
    )
    mlm_layer12: Optional[bool] = field(
        default=False, metadata={"help": "Use mlm loss on layer 12"}
    )
    entropy_layer2: Optional[bool] = field(
        default=False, metadata={"help": "Use entropy loss on layer 2"}
    )
    entropy_layer4: Optional[bool] = field(
        default=False, metadata={"help": "Use entropy loss on layer 4"}
    )
    entropy_layer6: Optional[bool] = field(
        default=False, metadata={"help": "Use entropy loss on layer 6"}
    )
    entropy_layer8: Optional[bool] = field(
        default=False, metadata={"help": "Use entropy loss on layer 8"}
    )
    entropy_layer10: Optional[bool] = field(
        default=False, metadata={"help": "Use entropy loss on layer 10"}
    )
    entropy_layer12: Optional[bool] = field(
        default=False, metadata={"help": "Use entropy loss on layer 12"}
    )
    use_entropy_layer: Optional[bool] = field(
        default=False, metadata={"help": "Use mlm loss on layer 12"}
    )
    task_name: Optional[str] = field(
        default='snli', metadata={"help": "Task to finetune - snli or mnli"}
    )
    batch_size: Optional[int] = field(
        default=32, metadata={"help": "Per device batch size"}
    )
    eval_batch_size: Optional[int] = field(
        default=128, metadata={"help": "Per device batch size"}
    )
    learning_rate: Optional[float] = field(
        default=2e-5, metadata={"help": "Initial learning rate"}
    )
    num_train_epochs: Optional[int] = field(
        default=20, metadata={"help": "Number of training epochs"}
    )
    exp_name: Optional[str] = field(
        default='exp0', metadata={"help": "Experiment name for wandb log"}
    )
        
        
from datacollator import DataCollatorForSim
from tokenizer import SimbertTokenizer
from model import SimbertForPreTraining, SimConfig
from datasets import concatenate_datasets

import torch

if __name__ == "__main__":
    n_gpus = max(1, torch.cuda.device_count())
    print("Found {} GPUs".format(n_gpus))
    parser = HfArgumentParser((PreprocessArguments,))
    model_args, = parser.parse_args_into_dataclasses()
    
    set_seed(42)
    tokenizer = SimbertTokenizer.from_pretrained(model_args.model_name, cache_dir=model_args.cache_dir)
    entropy = False
    if model_args.entropy_layer2 or model_args.entropy_layer4 or model_args.entropy_layer6 or model_args.entropy_layer8 or model_args.entropy_layer10 or model_args.entropy_layer12:
        entropy = True
    if entropy:
        if model_args.mlm_layer2 or model_args.mlm_layer4 or model_args.mlm_layer6 or model_args.mlm_layer8 or model_args.mlm_layer10 or model_args.mlm_layer12:
            print("Usage of mlm and entropy together is not supported")
    data_collator = DataCollatorForSim(tokenizer=tokenizer, entropy=entropy)
    
    config = SimConfig.from_pretrained(model_args.model_name, cache_dir=model_args.cache_dir)
    config.mlm_layer2 = model_args.mlm_layer2
    config.mlm_layer4 = model_args.mlm_layer4
    config.mlm_layer6 = model_args.mlm_layer6
    config.mlm_layer8 = model_args.mlm_layer8
    config.mlm_layer10 = model_args.mlm_layer10
    config.mlm_layer12 = model_args.mlm_layer12
    config.entropy_layer2 = model_args.entropy_layer2
    config.entropy_layer4 = model_args.entropy_layer4
    config.entropy_layer6 = model_args.entropy_layer6
    config.entropy_layer8 = model_args.entropy_layer8
    config.entropy_layer10 = model_args.entropy_layer10
    config.entropy_layer12 = model_args.entropy_layer12
    config.use_entropy_layer = model_args.use_entropy_layer
    config.num_labels = 3

    model = SimbertForPreTraining.from_pretrained(model_args.model_name, config=config, cache_dir=model_args.cache_dir)
    
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
        output_dir=os.path.join(model_args.output_dir, '{}_train_logs'.format(model_args.task_name)),
        per_device_train_batch_size=model_args.batch_size // n_gpus,
        per_device_eval_batch_size=model_args.eval_batch_size // n_gpus,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy='epoch',
        num_train_epochs=model_args.num_train_epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        seed=42,
        learning_rate=model_args.learning_rate,
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

