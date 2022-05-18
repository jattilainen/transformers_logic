import os
import sys
print(sys.prefix)
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import datasets
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

def preprocess_shuffled(examples):
    x = tokenizer(examples["shuffled_hypothesis"], examples["shuffled_premise"])
    return x


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

def compute_shuffled_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    if shuffled:
        with open(os.path.join(model_args.output_dir, 'predictions.npy'), 'rb') as f:
            norm_predictions = np.load(f)
        interesting_predictions = predictions[norm_predictions == labels]
        interesting_labels = labels[norm_predictions == labels]
        return {'accuracy': metric.compute(predictions=interesting_predictions, references=interesting_labels), 'neutral': (predictions == 1).mean()}
    else:
        with open(os.path.join(model_args.output_dir, 'predictions.npy'), 'wb') as f:
            np.save(f, predictions)
        return {'accuracy': metric.compute(predictions=predictions, references=labels), 'neutral': (predictions == 1).mean()}

    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

@dataclass
class PreprocessArguments:
    model_name: Optional[str] = field(
        default='roberta-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    datasets_dir: Optional[str] = field(
        default='datasets', metadata={"help": "Where do you want to store the datasets"}
    )
    cache_dir: Optional[str] = field(
        default='/home/vapavlov_4/.cache', metadata={"help": "Cache dir"}
    )
    output_dir: Optional[str] = field(
        default='exps/exp0', metadata={"help": "Where do you want to store results"}
    )
    base_model: Optional[str] = field(
        default='roberta-base', metadata={"help": "Model before SNLI/MNLI finetune"}
    )
    eval_batch_size: Optional[int] = field(
        default=128, metadata={"help": "Per device batch size"}
    )

import torch
import numpy as np
from torch import nn
from tqdm.notebook import tqdm
def calculate_intrasim(model, test_dataset, tokenizer):
    intra_sims = []
    random_sims = [[] for i in range(13)]
    random_prob = 0.003
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    for i in range(len(test_dataset)):
        inputs = tokenizer(test_dataset[i]['hypothesis'], test_dataset[i]['premise'], return_tensors="pt")
        hidden_states = model(**inputs.to(device))['hidden_states']
        example_sims = []
        for layer in range(len(hidden_states)):
            hs = hidden_states[layer].cpu()
            probs = torch.ones(hs.size()[1]) * random_prob
            taken = torch.bernoulli(probs)
            for i in range(len(taken)):
                if taken[i]:
                    random_sims[layer].append(hs[0, i])
            example_sims.append(cos(hs, hs.mean(dim=1, keepdim=True)).mean().item())
          
        intra_sims.append(example_sims)
    intra_sims = np.array(intra_sims)
    
    # calculate anisotropy
    simple_cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    layers_anisotropy = []
    for layer in range(13):
        ans = 0
        for i in range(len(random_sims[layer])):
            for j in range(i + 1, len(random_sims[layer])):
                ans += simple_cos(random_sims[layer][i], random_sims[layer][j]).item()
        ans /= (len(random_sims[layer]) * (len(random_sims[layer]) - 1)) / 2
        layers_anisotropy.append(ans)
        
    layer_sims = intra_sims.mean(axis=0)
    return layer_sims, np.array(layers_anisotropy)

if __name__ == "__main__":
    n_gpus = max(1, torch.cuda.device_count())
    print("Found {} GPUs".format(n_gpus))
    parser = HfArgumentParser((PreprocessArguments,))
    model_args, = parser.parse_args_into_dataclasses()
    
    metric = load_metric("accuracy")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, cache_dir=model_args.cache_dir)
    try:
        os.remove(os.path.join(model_args.output_dir, 'human_results.txt')) 
    except OSError:
        pass


    # shuffled
    for task_name in ['snli', 'mnli']:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        if task_name == 'snli':
            snli = datasets.load_from_disk(os.path.join(model_args.datasets_dir, 'shuffle_snli'))
            normal_snli = snli.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=["shuffled_premise", "shuffled_hypothesis", "premise", "hypothesis"])
            shuffled_snli = snli.map(preprocess_shuffled, batched=True).map(fix_negative, batched=True, remove_columns=["shuffled_premise", "shuffled_hypothesis", "premise", "hypothesis"])
            test_datasets = [("test", normal_snli["test"], shuffled_snli["test"])]
        if task_name == 'mnli':
            mnli = datasets.load_from_disk(os.path.join(model_args.datasets_dir, 'shuffle_mnli'))
            # not typo in preprocess_snli, because they both have hypothesis and premise
            normal_mnli = mnli.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=["shuffled_premise", "shuffled_hypothesis", 'promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'])
            shuffled_mnli = mnli.map(preprocess_shuffled, batched=True).map(fix_negative, batched=True, remove_columns=["shuffled_premise", "shuffled_hypothesis", 'promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'])
            test_datasets = [("validation_matched", normal_mnli["validation_matched"], shuffled_mnli["validation_matched"]), ("validation_mismatched", normal_mnli["validation_mismatched"], shuffled_mnli["validation_mismatched"])]

        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_args.output_dir, '{}_{}_final_model'.format(task_name, test_datasets[0][0])), num_labels=3, cache_dir=model_args.cache_dir)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(model_args.output_dir, 'human_eval_logs'),
            per_device_eval_batch_size=model_args.eval_batch_size // n_gpus,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy='epoch',
            seed=42,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_shuffled_metrics
        )

        with open(os.path.join(model_args.output_dir, 'human_results.txt'), 'a') as f:
            for test_name, test_dataset, test_shuffled in test_datasets:
                shuffled = False
                res = trainer.predict(test_dataset)
                normal_neutral = res.metrics['test_neutral']
                shuffled = True
                res = trainer.predict(test_shuffled)
                if trainer.is_world_process_zero():
                    print(task_name, test_name, res.metrics['test_accuracy'], normal_neutral, res.metrics['test_neutral'], file=f)
                    
                    
#     adv / con
    for task_name in ['snli', 'mnli']:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        if task_name == 'snli':
            adv_snli = datasets.load_from_disk(os.path.join(model_args.datasets_dir, 'adv_snli'))
            adv_snli = adv_snli.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=["premise", "hypothesis"])
            con_snli = datasets.load_from_disk(os.path.join(model_args.datasets_dir, 'contrastive_snli'))
            con_snli = con_snli.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=["premise", "hypothesis"])
            test_datasets = [("adv_snli", adv_snli["test"]), ("con_snli", con_snli["test"])]
            model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_args.output_dir, 'snli_test_final_model'), num_labels=3, cache_dir=model_args.cache_dir)
        if task_name == 'mnli':
            adv_mnli_matched = datasets.load_from_disk(os.path.join(model_args.datasets_dir, 'adv_mnli_matched'))
            adv_mnli_mismatched = datasets.load_from_disk(os.path.join(model_args.datasets_dir, 'adv_mnli_mismatched'))
            adv_mnli_matched = adv_mnli_matched.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=["premise", "hypothesis"])
            adv_mnli_mismatched = adv_mnli_mismatched.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=["premise", "hypothesis"])
            
            con_mnli_matched = datasets.load_from_disk(os.path.join(model_args.datasets_dir, 'contrastive_mnli_matched'))
            con_mnli_mismatched = datasets.load_from_disk(os.path.join(model_args.datasets_dir, 'contrastive_mnli_mismatched'))
            con_mnli_matched = con_mnli_matched.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=["premise", "hypothesis"])
            con_mnli_mismatched = con_mnli_mismatched.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=["premise", "hypothesis"])
            
            test_datasets = [("adv_mnli_matched", adv_mnli_matched["test"]), ("adv_mnli_mismatched", adv_mnli_mismatched["test"]), ("con_mnli_matched", con_mnli_matched["test"]), ("con_mnli_mismatched", con_mnli_mismatched["test"])]
            model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_args.output_dir, 'mnli_validation_matched_final_model'), num_labels=3, cache_dir=model_args.cache_dir)

        training_args = TrainingArguments(
            output_dir=os.path.join(model_args.output_dir, 'human_eval_logs'),
            per_device_eval_batch_size=model_args.eval_batch_size // n_gpus,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy='epoch',
            seed=42,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        with open(os.path.join(model_args.output_dir, 'human_results.txt'), 'a') as f:
            for test_name, test_dataset in test_datasets:
                res = trainer.predict(test_dataset)
                if trainer.is_world_process_zero():
                    print(test_name, res.metrics['test_accuracy'], file=f)

    # contextualization
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if model_args.base_model != 'roberta-base':
        base_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_args.output_dir, model_args.base_model), num_labels=3, cache_dir=model_args.cache_dir, output_hidden_states=True)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=3, cache_dir=model_args.cache_dir, output_hidden_states=True)
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_args.output_dir, 'snli_test_final_model'), num_labels=3, cache_dir=model_args.cache_dir, output_hidden_states=True)
    snli = load_dataset("snli", cache_dir=model_args.cache_dir)
    test_datasets = [('snli_test', snli['test'])]
    for test_name, test_dataset in test_datasets:
        model.to(device)
        model.eval()
        with torch.no_grad():
            intra_sim, anisotropy = calculate_intrasim(model, test_dataset, tokenizer)
        model.to('cpu')
        base_model.to(device)
        base_model.eval()
        with torch.no_grad():
            base_intra_sim, base_anisotropy = calculate_intrasim(base_model, test_dataset, tokenizer)
        base_model.to('cpu')
        with open(os.path.join(model_args.output_dir, 'human_results.txt'), 'a') as f:
            print(test_name, "intra_sim:", *intra_sim, file=f)
            print(test_name, "anisotropy:", *anisotropy, file=f)
            print(test_name, "base_intra_sim:", *base_intra_sim, file=f)
            print(test_name, "base_anisotropy:", *base_anisotropy, file=f)
            
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_args.output_dir, 'mnli_validation_matched_final_model'), num_labels=3, cache_dir=model_args.cache_dir, output_hidden_states=True)
    model.to(device)
    mnli = load_dataset("multi_nli", cache_dir=model_args.cache_dir)
    test_datasets = [('mnli_matched', mnli['validation_matched']), ('mnli_mismatched', mnli['validation_mismatched'])]
    for test_name, test_dataset in test_datasets:
        model.to(device)
        model.eval()
        with torch.no_grad():
            intra_sim, anisotropy = calculate_intrasim(model, test_dataset, tokenizer)
        model.to('cpu')
        base_model.to(device)
        base_model.eval()
        with torch.no_grad():
            base_intra_sim, base_anisotropy = calculate_intrasim(base_model, test_dataset, tokenizer)
        base_model.to('cpu')
        with open(os.path.join(model_args.output_dir, 'human_results.txt'), 'a') as f:
            print(test_name, "intra_sim:", *intra_sim, file=f)
            print(test_name, "anisotropy:", *anisotropy, file=f)
            print(test_name, "base_intra_sim:", *base_intra_sim, file=f)
            print(test_name, "base_anisotropy:", *base_anisotropy, file=f)
