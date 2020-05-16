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
from models.bert_crf_ner_model import BertCRFNerModel
from models.bert_bilstm_ner_model import BertBiLstmNerModel
from trainer.ner_trainer import NerTrainer
from utils import set_seed


# logger = logging.getLogger(__name__)

def main(args):
    # set random seed
    set_seed(args.seed)
    # load tokenizer
    do_lower_case = True
    if "bert_base_cased" in args.bert_model_dir:
        do_lower_case = False
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir, do_lower_case=do_lower_case)
    # load label map
    with open(args.label_map_path) as fin:
        label_list = [row.split()[0] for row in fin]

        label_list = ["PAD", "X"] + label_list  # for padding and subword
        # label_list = ["PAD"] + label_list  # for padding and subword
    label2idx = dict()
    idx2label = dict()
    for idx, label in enumerate(label_list):
        label2idx[label] = idx
        idx2label[idx] = label
    eval_labels = ['B-Term', 'I-Term', 'B-Definition', 'I-Definition', 'B-Alias-Term', 'I-Alias-Term',
                   'B-Referential-Term', 'I-Referential-Term', 'B-Referential-Definition', 'I-Referential-Definition',
                   'B-Qualifier', 'I-Qualifier']

    # load train and dev dataset
    train_dataset = NerSentDataset(dataset_path=args.train_dataset, tokenizer=tokenizer, label2idx=label2idx,
                                   max_seq_len=args.max_seq_len)
    dev_dataset = NerSentDataset(dataset_path=args.dev_dataset, tokenizer=tokenizer, label2idx=label2idx,
                                 max_seq_len=0)

    # load model
    if args.crf:
        model = BertCRFNerModel(args.bert_model_dir, n_labels=len(label_list))

    else:
        if args.bilstm:
            model = BertBiLstmNerModel(args.bert_model_dir, n_labels=len(label_list), loss_type=args.loss_type,
                                       label_pad_id=label2idx["PAD"])
        else:
            model = BertNerModel(args.bert_model_dir, n_labels=len(label_list), loss_type=args.loss_type,
                                 label_pad_id=label2idx["PAD"], gamma=args.gamma)

    trainer = NerTrainer(model=model, args=args, train_dataset=train_dataset, dev_dataset=dev_dataset,
                         tokenizer=tokenizer, label2idx=label2idx, idx2label=idx2label, eval_labels=eval_labels)

    trainer.train(last_epoch=0)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_dir", type=str, default="/home1/sxy/models/BERT/bert_base_uncased",
                        help="path to bert model and tokenizer")
    parser.add_argument("--model_dir", type=str, default="/home1/sxy/models/AdvancedNLP/checkpoints/para/tmp",
                        help="path to save model checkpoint")
    parser.add_argument("--log_dir", type=str, default="/home1/sxy/models/AdvancedNLP/logs",
                        help="path to output tensorboard log")
    parser.add_argument("--save_interval", type=int, default=1, help="interval to save checkpoints")
    parser.add_argument("--train_dataset", type=str,
                        default="/home1/sxy/datasets/AdvancedNLP/Data/train_and_dev/train.para",
                        help="path to train dataset")
    parser.add_argument("--dev_dataset", type=str,
                        default="/home1/sxy/datasets/AdvancedNLP/Data/train_and_dev/dev.para",
                        help="path to dev dataset")
    parser.add_argument("--label_map_path", type=str,
                        default="/home1/sxy/datasets/AdvancedNLP/Data/train_and_dev/label_map.txt",
                        help="path to label map file")
    parser.add_argument("--max_seq_len", type=int,
                        default=256, help="max input sequence length (after tokenizing)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--linear_lr", type=float, default=5e-4)
    parser.add_argument("--classifier_diff_lr", action='store_true')
    parser.add_argument("--warmup_proportion",
                        default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--clip_grad", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--loss_type", default="ce", type=str,
                        help="ce: CrossEntropy, focal: Focal Loss, smoothing: Label Smoothing")
    parser.add_argument("--gamma", default=2, type=float, help="gamma of focal loss")
    parser.add_argument("--froze_bert", action='store_true')
    parser.add_argument("--crf", action='store_true',  help="use crf layer or not")
    parser.add_argument("--bilstm", action='store_true', help="use bilstm layer or not")
    parser.add_argument("--fix_lr", action='store_true', help="if true, do not use lr schedule")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
