# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
sys.path.append(os.path.join(os.getcwd(), 'pytorch-pretrained-BERT'))

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert_local.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert_local.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert_local.tokenization import BertTokenizer
from pytorch_pretrained_bert_local.optimization import BertAdam, WarmupLinearSchedule

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, block=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.block = block


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_txt(cls, input_file):
        """Reads a text file."""
        return open(input_file).readlines()

class SstWikiProcessor(DataProcessor):
    """Processor for the SST-binary + Wikipedia data set (sentence level)."""

    def get_train_examples(self, data_dir, limit=0):
        """See base class."""
        ret = self._create_examples(
            SstWikiProcessor._read_txt(os.path.join(data_dir, "train.txt")), "train")
        if limit:
            ret = ret[0:limit]
        return ret

    def get_dev_examples(self, data_dir, limit=0):
        """See base class."""
        ret = self._create_examples(
            SstWikiProcessor._read_txt(os.path.join(data_dir, "dev.txt")), "dev")
        if limit:
            ret = ret[0:limit]
        return ret


    def get_test_examples(self, data_dir, limit=0):
        """See base class."""
        ret = self._create_examples(
            SstWikiProcessor._read_txt(os.path.join(data_dir, "test.txt")), "test")
        if limit:
            ret = ret[0:limit]
        return ret

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            label_text = line.split('\t', 1)
            text_split = label_text[1].split('[SEP]')
            text_sst = text_split[0].strip()
            text_wiki = text_split[1].strip() 
            label = label_text[0]
            examples.append(
                InputExample(guid=guid, text_a=text_wiki, text_b=text_sst, label=label))
        return examples

class PronounProcessor(DataProcessor):
    """Processor for the Pronoun data set (sentence level)."""

    def get_train_examples(self, data_dir, limit=0):
        """See base class."""
        ret = self._create_examples(
            PronounProcessor._read_txt(os.path.join(data_dir, "train.txt")),
            PronounProcessor._read_txt(os.path.join(data_dir, "train.txt.block")),
            "train")
        if limit:
            ret = ret[0:limit]
        return ret

    def get_dev_examples(self, data_dir, limit=0):
        """See base class."""
        ret = self._create_examples(
            PronounProcessor._read_txt(os.path.join(data_dir, "dev.txt")),
            PronounProcessor._read_txt(os.path.join(data_dir, "dev.txt.block")),
            "dev")
        if limit:
            ret = ret[0:limit]
        return ret


    def get_test_examples(self, data_dir, limit=0):
        """See base class."""
        ret = self._create_examples(
            PronounProcessor._read_txt(os.path.join(data_dir, "test.txt")),
            PronounProcessor._read_txt(os.path.join(data_dir, "test.txt.block")),
            "test")
        if limit:
            ret = ret[0:limit]
        return ret

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, block_lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            label_text = line.split('\t', 1)
            label = label_text[0]
            text = label_text[1]
            block = block_lines[i]
            examples.append(
                InputExample(guid=guid, text_a=text, text_b=None, label=label, block=block))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        if example.block:
            #segment_ids = [int(item) for item in example.block.split()]
            pronoun_list = ["her", "his", "him", "she", "he", "herself", "himself", "hers", "mr", "mrs", "ms", "mr.", "mrs.", "ms."]
            segment_ids = [1 if token.lower() in pronoun_list else 0 for token in tokens_a]
        else:
            segment_ids = [0]*len(tokens_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                segment_ids = segment_ids[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] + segment_ids + [0]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        # print("OOOOOOOO", len(input_ids), len(input_mask), len(segment_ids), max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(input_processor_type, preds, labels):
    assert len(preds) == len(labels)
    if input_processor_type == "sst-wiki" or input_processor_type == "pronoun":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(input_processor_type)

def attention_regularization_loss(attention_probs_layers, 
                                    pay_attention_mask,
                                    pad_attention_mask,
                                    hammer_coeff=0.0,
                                    optimize_func='mean',
                                    debug=False):
    float_type = torch.FloatTensor
    if torch.cuda.is_available():
        float_type = torch.cuda.FloatTensor

    reg_attention_mask = pay_attention_mask.unsqueeze(1).unsqueeze(2).type(float_type)
    pad_attention_mask = (1-pad_attention_mask).unsqueeze(1).unsqueeze(2).type(float_type)
    non_reg_attention_mask = 1 - (reg_attention_mask + pad_attention_mask)
    # attention_probs_layers - [B x H x aW x bW] [32, 12, 128, 128]
    # pay_attention_mask     -  B x W             32,  1,   1, 128
    #                        -  0..., 1..., 0... - WIKI, SST, PAD
    # minimize attention to SST words

    # We are only interested in last layer, and CLS token (first token)
    attention_probs_layer = attention_probs_layers[-1][:, :, 0, :].unsqueeze(2)
    # 32, 12, 1, 128

    reg_attention_maps     = attention_probs_layer * reg_attention_mask
    pad_attention_maps     = attention_probs_layer * pad_attention_mask
    non_reg_attention_maps = attention_probs_layer * non_reg_attention_mask
    if debug:
        print(f"Regularized attention mask:{reg_attention_mask}")
        print(f"Non-Regular attention mask:{non_reg_attention_mask}")
    # 32, 12, 1, 128
    # 32, 12, 1, 128 -> 32, 12, 1
    reg_attention_sum = torch.sum(reg_attention_maps, -1)
    pad_attention_sum = torch.sum(pad_attention_maps, -1)
    non_reg_attention_sum = torch.sum(non_reg_attention_maps, -1)

    if optimize_func == 'mean':
        hammer_reg = torch.mean( torch.log(1 - reg_attention_sum) )
    else:
        # minimize max attention_sum
        # minimize min log(1 - attention_sum)
        hammer_reg = torch.min( torch.log(1 - reg_attention_sum) )

    return - hammer_coeff * hammer_reg, torch.mean(reg_attention_sum), torch.mean(non_reg_attention_sum), torch.mean(pad_attention_sum), torch.max(reg_attention_sum)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--input_processor_type",
                        default="sts-b",
                        type=str,
                        required=True,
                        help="The type of processor to use for reading data.")
    parser.add_argument("--output_dir",
                        default="sst-wiki-output",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--hammer_coeff',
                        type=float,
                        default=0.0,
                        help="Hammer loss coefficient")
    parser.add_argument('--att_opt_func',
                        type=str,
                        default="mean",
                        help="Attention optimization function")
    parser.add_argument("--debug",
                        action='store_true')
    parser.add_argument("--first_run",
                        action='store_true')
    parser.add_argument("--name",
                        type=str)
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    base_labels = {}
    print(f"FIRST RUN: {args.first_run}")
    if not args.first_run:
        for typ in ["dev", "test"]:
            base_labels_content = open("{}_base_labels_{}.txt".format(args.name, typ), 'r').readlines()
            base_labels[typ] = [int(label.strip()) for label in base_labels_content]
            
    debug = args.debug
    if debug:
        args.train_batch_size = 2
        args.eval_batch_size = 2
        args.num_train_epochs = 1

    processors = {
        "sst-wiki": SstWikiProcessor,
        "pronoun": PronounProcessor
    }

    output_modes = {
        "sst-wiki": "classification",
        "pronoun": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_processor_type = args.input_processor_type.lower()

    if input_processor_type not in processors:
        raise ValueError("Task not found: %s" % (input_processor_type))

    processor = processors[input_processor_type]()
    output_mode = output_modes[input_processor_type]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        limit = 2 if debug else 0
        train_examples = processor.get_train_examples(args.data_dir, limit)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
              cache_dir=cache_dir,
              num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        print("typ\tepoch\tacc\tavg_mean_mass\tavg_max_mass\tloss\thammer_loss\tlabel_match_score\tavg_mean_vn\tavg_max_vn\tavg_min_vn")
        model.train()

        for epoch in trange(int(args.num_train_epochs) + 1, desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            if epoch > 0:
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch

                    # define a new function to compute loss values for both output_modes
                    logits, attention_probs_layers, category_mask, _ = model(input_ids, 
                                                                            token_type_ids=segment_ids,
                                                                            pad_attention_mask=input_mask,
                                                                            manipulate_attention=True,
                                                                            category_mask=None,
                                                                            labels=None)
                    # logits - B x 2 
                    loss_fct = CrossEntropyLoss() # averages the loss over B
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                    loss += attention_regularization_loss(attention_probs_layers, 
                                                            category_mask,
                                                            input_mask,
                                                            args.hammer_coeff, 
                                                            optimize_func=args.att_opt_func,
                                                            debug=debug)[0]
                    
                    if n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            # modify learning rate with special warm up BERT uses
                            # if args.fp16 is False, BertAdam is used that handles this automatically
                            lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step/num_train_optimization_steps,
                                                                                    args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                    if debug:
                        break

            # EVALUATION AFTER EVERY EPOCH
            eval_preds = {}
            for typ in ["dev", "test"]:
                eval_preds[typ] = run_evaluation(args, processor, label_list, tokenizer, output_mode, epoch, 
                                                    model, num_labels, tr_loss, global_step, device, input_processor_type, 
                                                    base_labels, debug, typ)

            #dump labels after the last epoch, or when first_run
            if args.first_run or epoch == args.num_train_epochs:
                for typ in ["dev", "test"]:
                    preds = eval_preds[typ]
                    filename = "{}_labels_{}_.txt".format(typ, epoch)
                    labels_file = os.path.join(args.output_dir, filename)
                    with open(labels_file, "w") as writer:
                        logger.info("Dumping labels in the file: {}".format(labels_file))
                        writer.write('\n'.join([str(pred) for pred in preds]))

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)

def run_evaluation(args, processor, label_list, tokenizer, output_mode, epoch, 
                    model, num_labels, tr_loss, global_step, device, input_processor_type, 
                    base_labels, debug, typ="dev"):
    
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        
        limit = 2 if debug else 0
        if typ == "dev":
            eval_examples = processor.get_dev_examples(args.data_dir, limit)
        else:
            eval_examples = processor.get_test_examples(args.data_dir, limit)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation on " + typ + " data*****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        ce_eval_loss = 0
        ar_eval_loss = 0
        nb_eval_steps = 0
        preds = []

        tmp_avg_attention_mass = 0.0
        tmp_max_attention_mass = 0.0
        tmp_non_reg_mass = 0.0
        tmp_pad_mass = 0.0

        tmp_vnfs = [0., 0., 0.]

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits, attention_probs_layers, category_mask, vnfs = model(input_ids, 
                                                                        token_type_ids=segment_ids,
                                                                        pad_attention_mask=input_mask,
                                                                        manipulate_attention=True,
                                                                        category_mask=None, 
                                                                        labels=None)

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss() # averages the loss over B
            tmp_ce_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            tmp_ar_eval_loss, avg_attention_mass, non_reg_mass, pad_mass, max_attention_mass = \
                attention_regularization_loss(attention_probs_layers, 
                                                category_mask,
                                                input_mask,
                                                args.hammer_coeff,
                                                optimize_func=args.att_opt_func)

            tmp_avg_attention_mass += avg_attention_mass.item()
            tmp_max_attention_mass += max_attention_mass.item()
            tmp_non_reg_mass += non_reg_mass.item()
            tmp_pad_mass += pad_mass.item()

            tmp_vnfs = [tmp_vnfs[i] + vnfs[i] for i in range(3)]

            tmp_eval_loss = tmp_ce_eval_loss + tmp_ar_eval_loss

            eval_loss += tmp_eval_loss.mean().item()
            ce_eval_loss += tmp_ce_eval_loss.mean().item()
            ar_eval_loss += tmp_ar_eval_loss.mean().item()

            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

            if debug:
                break
        eval_loss = eval_loss / nb_eval_steps
        ce_eval_loss = ce_eval_loss / nb_eval_steps
        ar_eval_loss = ar_eval_loss / nb_eval_steps

        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(input_processor_type, preds, all_label_ids.numpy())
        loss = tr_loss/global_step if (args.do_train and epoch > 0) else None

        result['eval_loss'] = eval_loss
        result['ce_eval_loss'] = ce_eval_loss
        result['ar_eval_loss'] = ar_eval_loss

        result['global_step'] = global_step
        result['loss'] = loss

        result['avg_mean_attention_mass'] = tmp_avg_attention_mass / nb_eval_steps
        result['avg_max_attention_mass'] = tmp_max_attention_mass / nb_eval_steps
        result['avg_non_reg_attention_mass'] = tmp_non_reg_mass / nb_eval_steps
        result['avg_pad_attention_mass'] = tmp_pad_mass / nb_eval_steps

        result['avg_mean_value_norm'] = tmp_vnfs[0]*1. / nb_eval_steps
        result['avg_max_value_norm'] = tmp_vnfs[1]*1. / nb_eval_steps
        result['avg_min_value_norm'] = tmp_vnfs[2]*1. / nb_eval_steps

        result['label_match_score'] = 0.0
        if not args.first_run:
            num_labels = len(preds)
            result['label_match_score'] = simple_accuracy(preds, base_labels[typ][0:num_labels])

        output_eval_file = os.path.join(args.output_dir, typ + "_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** {} results *****".format(typ))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        print('\t'.join([ str(elem) for elem in 
                        [typ, 
                        epoch, 
                        result['acc'], 
                        result['avg_mean_attention_mass'],
                        result['avg_max_attention_mass'],
                        result['eval_loss'], 
                        result['ar_eval_loss'],
                        result['label_match_score'],
                        result['avg_mean_value_norm'], 
                        result['avg_max_value_norm'], 
                        result['avg_min_value_norm'] 
                    ]]))
        
        return preds


if __name__ == "__main__":
    main()
