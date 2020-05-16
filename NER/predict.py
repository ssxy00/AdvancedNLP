# -*- coding: utf-8 -*-
# @Time        : 2020/5/5 23:44
# @Author      : ssxy00
# @File        : predict.py
# @Description : 给定待预测的输入文件和模型，生成输出文件

import os
import argparse
from tqdm import tqdm
from transformers import BertTokenizer

import torch

from models.bert_ner_model import BertNerModel
from utils import set_seed

def predict(model, input_ids, require_labels):
    with torch.no_grad():
        label_indices = model(input_ids).argmax(-1)
        return label_indices.masked_select(require_labels).cpu().numpy().tolist()



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


    # load model
    device = torch.device(args.device)
    model = BertNerModel(args.bert_model_dir, n_labels=len(label_list))
    state_dict = torch.load(args.checkpoint_path)
    model.load_state_dict(state_dict['model'], strict=False)


    # eval
    model.to(device)
    model.eval()


    file_names = os.listdir(args.eval_dir)
    file_names.sort()
    for file_name in tqdm(file_names):
        if not '.deft' in file_name:
            print(file_name)
        eval_path = os.path.join(args.eval_dir, file_name)
        output_path = os.path.join(args.output_dir, file_name.split('.')[0] + '_pred.deft')


        infos_of_sent = [] # 记录当前待预测的 sentence 每个 token 的信息
        index_tokens = [] # 记录组成句子的 index 序列 （分词后，index 后）
        require_labels = [] # 和 index_tokens 等长的序列，1 表示该位置需要输出 label，0 表示不需要 (subword)
        with open(eval_path) as fin, open(output_path, 'w') as fout:
            last_line_is_empty = False
            empty_line_count = 0
            for line in tqdm(fin):
                if not line.strip(): # 一个空行
                    if not last_line_is_empty:
                        last_line_is_empty = True
                        infos_of_sent.append("<empty-line>")
                    else: # 两个空行，分段
                        # do predict
                        index_tokens = [tokenizer.cls_token_id] + index_tokens + [tokenizer.sep_token_id]
                        require_labels = [0] + require_labels + [0]
                        index_tokens = torch.tensor([index_tokens], dtype=torch.long, device=device)
                        require_labels = torch.tensor(require_labels, dtype=torch.bool, device=device)
                        label_indices = predict(model=model, input_ids=index_tokens, require_labels=require_labels)
                        # output predict results
                        token_idx = 0
                        for info in infos_of_sent:
                            if info == "<empty-line>":
                                fout.write('\n')
                            else:
                                predict_label = label_list[label_indices[token_idx]]
                                token_idx += 1
                                # 如果预测 "X"，处理成 "O"
                                if predict_label == "X":
                                    predict_label = "O"
                                predict_line = '\t'.join(info + [predict_label])
                                fout.write(predict_line + '\n')
                        infos_of_sent = []
                        index_tokens = []
                        require_labels = []
                        fout.write('\n')
                        last_line_is_empty = True
                    continue
                else:
                    last_line_is_empty = False
                    token, source_file, start_char, end_char = line.split()[:4]
                    infos_of_sent.append([token, source_file, start_char, end_char])
                    sub_tokens = tokenizer.tokenize(token) # 分词
                    index_tokens += tokenizer.convert_tokens_to_ids(sub_tokens)
                    require_labels += ([1] + [0] * (len(sub_tokens) - 1))


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_dir", type=str, default="/home1/sxy/models/BERT/bert_base_uncased",
                        help="path to bert model and tokenizer")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/home1/sxy/models/AdvancedNLP/checkpoints/bert/para_level/1e-4/checkpoint3.pt",
                        help="model checkpoint to evaluate")
    parser.add_argument("--output_dir", type=str,
                        default="/home1/sxy/datasets/AdvancedNLP/Data/test/subtask_2_results",
                        help="file to output predicting results")
    parser.add_argument("--eval_dir", type=str,
                        default="/home1/sxy/datasets/AdvancedNLP/Data/test/subtask_2",
                        help="path to eval file")
    parser.add_argument("--label_map_path", type=str,
                        default="/home1/sxy/datasets/AdvancedNLP/Data/train_and_dev/label_map.txt",
                        help="path to label map file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()