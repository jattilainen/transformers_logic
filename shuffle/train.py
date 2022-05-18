# this file should be used to train cline on existing dataset
import random
from dataclasses import dataclass, field
from typing import Optional

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from datasets import Dataset
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
)

@dataclass
class PreprocessArguments:
    model_name: Optional[str] = field(
        default='roberta-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='/home/vapavlov_4/.cache', metadata={"help": "Cache dir"}
    )
    dataset_file: Optional[str] = field(
        default='/home/vapavlov_4/datasets/cline_snli', metadata={"help": "Where is the dataset"}
    )
    output_dir: Optional[str] = field(
        default='/home/vapavlov_4/models/robeta_base_cline_snli', metadata={"help": "Where do you want to store model"}
    )
    batch_size: Optional[int] = field(
        default=32, metadata={"help": "Per device batch size"}
    )
    predict_tokens: Optional[bool] = field(
        default=False, metadata={"help": "Predict shuffled tokens or only classify them whether shuffled or not"}
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
        
        
from datacollator import DataCollatorForShuffle
from tokenizer import ShuffleTokenizer
from model import ShuffleBertForPreTraining, ShuffleConfig
from datasets import concatenate_datasets

import torch

def preprocess_snli(examples):
    x = tokenizer(examples["hypothesis"])
    return x

def preprocess_news(examples):
    x = tokenizer(examples["text"])
    return x

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print("Found {} GPUs".format(n_gpus))
    n_gpus = max(1, n_gpus)
    parser = HfArgumentParser((PreprocessArguments,))
    model_args, = parser.parse_args_into_dataclasses()
    tokenizer = ShuffleTokenizer.from_pretrained(model_args.model_name, cache_dir=model_args.cache_dir)

    shuffle = True
    if '/' in model_args.dataset_file:
        shuffle = False
        dataset = datasets.load_from_disk(model_args.dataset_file)
    else:
        dataset = datasets.load_dataset(model_args.dataset_file, cache_dir=model_args.cache_dir)
        if 'hypothesis' in dataset['train'].features.keys():
            dataset = dataset.map(preprocess_snli, batched=True, remove_columns=["premise", "hypothesis", 'label'])
        else:
            dataset = dataset.map(preprocess_news, batched=True, remove_columns=["text", 'label'])

    set_seed(42)
    
    config = ShuffleConfig.from_pretrained(model_args.model_name, cache_dir=model_args.cache_dir)
    config.predict_tokens = model_args.predict_tokens
    if model_args.predict_tokens:
        config.num_labels = config.vocab_size
    else:
        config.num_labels = 2
    data_collator = DataCollatorForShuffle(tokenizer=tokenizer, shuffle=shuffle, predict_tokens=config.predict_tokens)

    model = ShuffleBertForPreTraining.from_pretrained(model_args.model_name, config=config, cache_dir=model_args.cache_dir)
    
    train_dataset = dataset["train"].shuffle(seed=42)
    if "vadidation" in dataset.keys():
        eval_dataset = dataset["validation"]
    else:
        eval_dataset = dataset["test"]
    
    training_args = TrainingArguments(
        output_dir=os.path.join(model_args.output_dir, 'shuffle'),
        per_device_train_batch_size=model_args.batch_size // n_gpus,
        per_device_eval_batch_size=model_args.eval_batch_size // n_gpus,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy='epoch',
        num_train_epochs=model_args.num_train_epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
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
        data_collator=data_collator,
    )
    
    trainer.train()
    if trainer.is_world_process_zero():
        trainer.save_model(os.path.join(model_args.output_dir, 'best_shuffle'))
