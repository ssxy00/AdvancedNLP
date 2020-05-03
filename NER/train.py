# -*- coding: utf-8 -*-
# @Time        : 2020/5/2 23:40
# @Author      : ssxy00
# @File        : train.py
# @Description :

import logging
import argparse
from transformers import BertTokenizer

import torch

from datasets.ner_sent_dataset import NerSentDataset
from models.bert_ner_model import BertNerModel
from trainer.ner_trainer import NerTrainer
from utils import set_seed


# logger = logging.getLogger(__name__)

def main(args):
    # set random seed
    set_seed(args.seed)
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
    # load label map
    with open(args.label_map_path) as fin:
        label_list = [row.split()[0] for row in fin]
        label_list = ["PAD", "X"] + label_list  # for padding and subword
    label2idx = dict()
    for idx, label in enumerate(label_list):
        label2idx[label] = idx

    # load train and dev dataset
    train_dataset = NerSentDataset(dataset_path=args.train_dataset, tokenizer=tokenizer, label2idx=label2idx,
                                   max_seq_len=args.max_seq_len)
    dev_dataset = NerSentDataset(dataset_path=args.dev_dataset, tokenizer=tokenizer, label2idx=label2idx,
                                 max_seq_len=args.max_seq_len)

    # load model
    model = BertNerModel(args.bert_model_dir, n_labels=len(label_list))

    trainer = NerTrainer(model=model, args=args, train_dataset=train_dataset, dev_dataset=dev_dataset,
                         tokenizer=tokenizer, label2idx=label2idx)

    trainer.train(last_epoch=0)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_dir", type=str, default="/home1/sxy/models/BERT/bert_base_uncased",
                        help="path to bert model and tokenizer")
    parser.add_argument("--model_dir", type=str, default="/home1/sxy/models/AdvancedNLP/checkpoints",
                        help="path to save model checkpoint")
    parser.add_argument("--log_dir", type=str, default="/home1/sxy/models/AdvancedNLP/logs",
                        help="path to output tensorboard log")
    parser.add_argument("--save_interval", type=int, default=1, help="interval to save checkpoints")
    parser.add_argument("--train_dataset", type=str,
                        default="/home1/sxy/datasets/AdvancedNLP/Data/train_and_dev/dev.sent", # TODO
                        help="path to train dataset")
    parser.add_argument("--dev_dataset", type=str,
                        default="/home1/sxy/datasets/AdvancedNLP/Data/train_and_dev/dev.sent",
                        help="path to dev dataset")
    parser.add_argument("--label_map_path", type=str,
                        default="/home1/sxy/datasets/AdvancedNLP/Data/train_and_dev/label_map.txt",
                        help="path to label map file")
    parser.add_argument("--max_seq_len", type=int,
                        default=128, help="max input sequence length (after tokenizing)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_proportion",
                        default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--clip_grad", default=1.0, type=float,
                        help="Max gradient norm.")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
