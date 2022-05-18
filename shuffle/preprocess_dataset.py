# this file should be used to create dataset for cline model from snli dataset

from nltk.corpus import wordnet as wn

import nltk
from nltk.corpus import gutenberg

import random
from dataclasses import dataclass, field
from typing import Optional

import spacy
spacy_nlp = spacy.load('en_core_web_sm')

from spacy.tokens import Doc
Doc.set_extension('_synonym_sent', default=False)
Doc.set_extension('_synonym_intv', default=False)
Doc.set_extension('_antonym_sent', default=False)
Doc.set_extension('_antonym_intv', default=False)

moby = set(nltk.Text(gutenberg.words('melville-moby_dick.txt')))
moby = [word.lower() for word in moby if len(word) >2]

REPLACE_RATIO = 0.5

REPLACE_ORIGINAL = 0
REPLACE_LEMMINFLECT = 1
REPLACE_SYNONYM = 2
REPLACE_HYPERNYMS = 3
REPLACE_ANTONYM = 4
REPLACE_RANDOM = 5
REPLACE_ADJACENCY = 6

REPLACE_NONE = -100

SYNONYM_RATIO = 1/3
HYPERNYMS_RATIO = 1/3
LEMMINFLECT_RATIO = 1/3

ANTONYM_RATIO = 1/2
RANDOM_RATIO = 1/2

import random

from wordnet import (
    REPLACE_POS,
    get_synonym,
    get_hypernyms,
    get_antonym,
    get_lemminflect
)

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from datasets import load_dataset
from transformers import AutoTokenizer

from transformers import (
    HfArgumentParser,
    PreTrainedTokenizer
)

@dataclass
class PreprocessArguments:
    model_name: Optional[str] = field(
        default='roberta-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='.cache', metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    output_file: Optional[str] = field(
        default='datasets/cline_snli', metadata={"help": "Where do you want preprocessed dataset"}
    )
    dataset_name: Optional[str] = field(
        default='snli', metadata={"help": "What dataset do you use"}
    )
        
        
def search_replacement(doc, candidate_index, replace_type, max_num, pos_to_words=None):
    sr_rep = []
    if max_num < 1:
        return sr_rep

    for r_idx in candidate_index:
        token = doc[r_idx]
        rep = None
        if replace_type == REPLACE_ANTONYM:
            reps = get_antonym(token)
            rep = random.choice(reps) if reps else None
        elif replace_type == REPLACE_ADJACENCY:
            reps = pos_to_words[token.pos_]
            rep = random.choice(reps) if reps else None
        elif replace_type == REPLACE_RANDOM:
            rep = moby[int(random.random()*len(set(moby)))]
        elif replace_type == REPLACE_SYNONYM:
            reps = get_synonym(token)
            rep = random.choice(reps) if reps else None
        elif replace_type == REPLACE_HYPERNYMS:
            reps = get_hypernyms(token)
            rep = random.choice(reps) if reps else None
        elif replace_type == REPLACE_LEMMINFLECT:
            reps = get_lemminflect(token)
            rep = random.choice(reps) if reps else None
        else:
            pass

        if rep and rep.lower() != token.text.lower():
            sr_rep.append((r_idx, rep, replace_type))

        if len(sr_rep) >= max_num:
            break
    return sr_rep


from spacy.language import Language

@Language.component('replace_word')
def replace_word(doc):
    synonym_sent = []
    synonym_intv = []
    ori_syn_intv = []
    antonym_sent = []
    antonym_intv = []
    ori_ant_intv = []

    length = len(doc)
    rep_num = int(length*REPLACE_RATIO)

    rep_index = []
    # pos_word = {p:[] for p in REPLACE_POS}
    for index, token in enumerate(doc):
        if token.pos_ in REPLACE_POS:
            rep_index.append(index)
            # pos_word[token.pos_].append(token.text)
    rep_num = min(rep_num, len(rep_index))

    syn_rand = random.random()
    ant_rand = random.random()

    syn_index = rep_index[:]
    random.shuffle(syn_index)
    ant_index = rep_index[:]
    random.shuffle(ant_index)
    syn_replace = []
    ant_replace = [] # [(rep_idx, rep_word, rep_type)]

    ############### Antonym Replacement ####################
    if ant_rand < ANTONYM_RATIO:
        ant_replace = search_replacement(doc, candidate_index=ant_index, replace_type=REPLACE_ANTONYM, max_num=rep_num)
    # if not ant_replace and ant_rand < ANTONYM_RATIO + ADJACENCY_RATIO:
    #     ant_replace = search_replacement(doc, candidate_index=ant_index, replace_type=REPLACE_ADJACENCY, max_num=rep_num, pos_to_words=pos_word)

    if not ant_replace:
        ant_replace = search_replacement(doc, candidate_index=ant_index, replace_type=REPLACE_RANDOM, max_num=rep_num)
    ############### Synonym Replacement ####################
    if syn_rand < HYPERNYMS_RATIO:
        syn_replace = search_replacement(doc, candidate_index=syn_index, replace_type=REPLACE_HYPERNYMS, max_num=rep_num)

    if not syn_replace and syn_rand < HYPERNYMS_RATIO + SYNONYM_RATIO:
        syn_replace = search_replacement(doc, candidate_index=syn_index, replace_type=REPLACE_SYNONYM, max_num=rep_num)

    if not syn_replace:
        syn_replace = search_replacement(doc, candidate_index=syn_index, replace_type=REPLACE_LEMMINFLECT, max_num=rep_num)
    ############### Original Replacement ####################

    all_replace = ant_replace + syn_replace
    all_replace = sorted(all_replace, key=lambda x:x[0], reverse=True)

    ori_len = -1 # point to the space before next token
    syn_len = -1
    ant_len = -1
    rep_idx, rep_word, rep_type = all_replace.pop() if all_replace else (None, None, None)
    for index, token in enumerate(doc):
        ori = syn = ant = token.text

        while index == rep_idx:
            if rep_type in [REPLACE_SYNONYM, REPLACE_HYPERNYMS, REPLACE_LEMMINFLECT]:
                syn = rep_word
                synonym_intv.append((syn_len, syn_len + len(syn.encode('utf-8')), rep_type)) # fix length mismatch, mx.encode for bytelevelbpe
            elif rep_type in [REPLACE_ANTONYM, REPLACE_RANDOM]:
                ant = rep_word
                antonym_intv.append((ant_len, ant_len + len(ant.encode('utf-8')), rep_type))
            else:
                pass

            rep_idx, rep_word, rep_type = all_replace.pop() if all_replace else (None, None, None)

        if index in rep_index:
            if ori == syn:
                synonym_intv.append((syn_len, syn_len + len(syn.encode('utf-8')), REPLACE_ORIGINAL))
            if ori == ant:
                antonym_intv.append((ant_len, ant_len + len(ant.encode('utf-8')), REPLACE_ORIGINAL))

        ori_len = ori_len + len(ori.encode('utf-8')) + 1
        syn_len = syn_len + len(syn.encode('utf-8')) + 1 # +1 to point the space before next token
        ant_len = ant_len + len(ant.encode('utf-8')) + 1

        synonym_sent.append(syn)
        antonym_sent.append(ant)

    doc._._synonym_sent = synonym_sent
    doc._._synonym_intv = synonym_intv
    doc._._antonym_sent = antonym_sent
    doc._._antonym_intv = antonym_intv

    return doc


def word_replace(examples):
    inputs = examples['text']

    original_sent = []
    synonym_sent = []
    synonym_intv = []
    antonym_sent = []
    antonym_intv = []
    docs = spacy_nlp.pipe(inputs, n_process=1, batch_size=100, disable=['parser', 'ner'])
    for doc in docs:
        ori_sent = " ".join([t.text for t in doc])
        syn_sent = " ".join(doc._._synonym_sent)
        ant_sent = " ".join(doc._._antonym_sent)

        syn_intv = doc._._synonym_intv
        ant_intv = doc._._antonym_intv

        original_sent.append(ori_sent)
        synonym_sent.append(syn_sent)
        synonym_intv.append(syn_intv)
        antonym_sent.append(ant_sent)
        antonym_intv.append(ant_intv)

    return {'original_sent': original_sent,
            'synonym_sent': synonym_sent,
            'synonym_intv': synonym_intv,
            'antonym_sent': antonym_sent,
            'antonym_intv': antonym_intv}
    

def get_replace_label(word_list, repl_intv, orig_sent):
    label = [REPLACE_NONE] * len(word_list)
    if not repl_intv:
        return label
    byte_index = 0 # point to the start of the next token in the byte type sentence
    orig_index = 0 # point to the start of the next token in the utf-8 type sentence
    cur_range = 0
    cur_start, cur_end, cur_label = repl_intv[cur_range] # raplacement range is of increasing ordered (include spaces in text)
    for index, word in enumerate(word_list):
        if byte_index >= cur_start and byte_index <= cur_end: # word piece is in replacement range
            label[index] = cur_label

        byte_offset = len(word) # bytelevel contains spaces in the token

        byte_index += byte_offset # bytelevel contains spaces in the token
        if byte_index > cur_end: # update replacement range
            if cur_range != len(repl_intv)-1: # not the last range
                cur_range += 1
                cur_start, cur_end, cur_label = repl_intv[cur_range]
            else: # no new range
                break

    assert cur_range == len(repl_intv)-1

    return label


def convert_tokens_to_ids(examples):
    input_ids = []
    ori_syn_label = []
    ori_ant_label = []
    synonym_ids = []
    synonym_label = []
    antonym_ids = []
    antonym_label = []

    exp_nums = len(examples['original_sent'])
    for i in range(exp_nums):
        ori_sent = tokenizer.tokenize(examples['original_sent'][i])
        syn_sent = tokenizer.tokenize(examples['synonym_sent'][i])
        ant_sent = tokenizer.tokenize(examples['antonym_sent'][i])

        syn_labl = get_replace_label(syn_sent, examples['synonym_intv'][i], examples['synonym_sent'][i])
        ant_labl = get_replace_label(ant_sent, examples['antonym_intv'][i], examples['antonym_sent'][i])

        ori_ids = tokenizer.convert_tokens_to_ids(ori_sent)
        syn_ids = tokenizer.convert_tokens_to_ids(syn_sent)
        ant_ids = tokenizer.convert_tokens_to_ids(ant_sent)

        input_ids.append(ori_ids)
        synonym_ids.append(syn_ids)
        synonym_label.append(syn_labl)
        antonym_ids.append(ant_ids)
        antonym_label.append(ant_labl)

    return {'input_ids': input_ids,
            'synonym_ids': synonym_ids,
            'synonym_label': synonym_label,
            'antonym_ids': antonym_ids,
            'antonym_label': antonym_label}


def preprocess_snli(examples):
    examples['text'] = examples["hypothesis"]
    return examples

def preprocess_news(examples):
    examples['text'] = examples["text"]
    return examples

if __name__ == "__main__":
    parser = HfArgumentParser((PreprocessArguments,))
    model_args, = parser.parse_args_into_dataclasses()
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, cache_dir=model_args.cache_dir)

    # get and prepare dataset
    if model_args.dataset_name == 'snli':
        snli = load_dataset("snli", cache_dir=model_args.cache_dir)
        cline_dataset = snli.map(preprocess_snli, batched=True, remove_columns=["premise", "hypothesis", 'label'])
    elif model_args.dataset_name == 'ag_news':
        news = load_dataset('ag_news', cache_dir=model_args.cache_dir)
        cline_dataset = news.map(preprocess_news, batched=True, remove_columns=['label'])
        
    spacy_nlp.add_pipe('replace_word', last=True)
    preprocess_cline = cline_dataset.map(word_replace,
                        batched=True,
                        remove_columns='text')
    preprocess_cline.set_format(type=None, columns=['original_sent', 'synonym_sent', 'synonym_intv', 'antonym_sent', 'antonym_intv'])
    
    preprocess_cline = preprocess_cline.map(convert_tokens_to_ids,
                            batched=True,
                            remove_columns=['original_sent', 'synonym_sent', 'synonym_intv', 'antonym_sent', 'antonym_intv'])

    preprocess_cline.set_format(type=None, columns=['input_ids', 'synonym_ids', 'synonym_label', 'antonym_ids', 'antonym_label'])
    
    preprocess_cline.save_to_disk(model_args.output_file)