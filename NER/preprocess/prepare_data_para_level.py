# -*- coding: utf-8 -*-
# @Time        : 2020/5/2 22:00
# @Author      : ssxy00
# @File        : prepare_data.py
# @Description : 将 train 或 dev 目录下的文件合并成一个文件

import os
from tqdm import tqdm

def get_paras_from_single_file(file_path):
    paras = []
    para = []
    last_line_is_empty = False
    with open(file_path) as fin:
        for row in fin:
            row = row.strip()
            if not row:
                if para and last_line_is_empty:
                    paras.append(' '.join(para))
                    para = []
                last_line_is_empty = True
            else:
                last_line_is_empty = False
                row = row.split()
                assert len(row) == 8
                para.append(row[0].strip() + '<###>' + row[4].strip())
        if para:
            paras.append(' '.join(para))
    return paras



def prepare_data(raw_dir, output_path):
    """
    :param raw_dir: train or dev file directory
    :param output_path: output file, each line is a sentence with format: word1<###>tag1 word2<###>tag2 ...
    """
    output_paras = []
    file_names = os.listdir(raw_dir)
    file_names.sort()
    for file_name in tqdm(file_names):
        if not '.deft' in file_name:
            print(file_name)
            continue

        paras_from_single_file = get_paras_from_single_file(os.path.join(raw_dir, file_name))
        output_paras += paras_from_single_file

    with open(output_path, 'w') as fout:
        for line in output_paras:
            fout.write(line + '\n')




if __name__ == "__main__":
    data_dir = "/home1/sxy/datasets/AdvancedNLP/Data/train_and_dev"
    type = "train"
    raw_dir = os.path.join(data_dir, type)
    output_path = os.path.join(data_dir, type + '.para')
    prepare_data(raw_dir, output_path)
