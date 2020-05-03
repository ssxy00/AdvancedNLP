# -*- coding: utf-8 -*-
# @Time        : 2020/5/3 10:40
# @Author      : ssxy00
# @File        : prepare_label_map.py
# @Description : 从训练数据中找出所有 labels

import os
from tqdm import tqdm

def get_lines_from_single_file(file_path):
    lines = []
    line = []
    with open(file_path) as fin:
        for row in fin:
            row = row.strip()
            if not row:
                if line:
                    lines.append(' '.join(line))
                    line = []
            else:
                row = row.split()
                assert len(row) == 8
                line.append(row[0].strip() + '###' + row[4].strip())
        if line:
            lines.append(' '.join(line))
    return lines



def generate_label_map(train_data_dir, label_map_path):
    """
    :param train_data_dir: train or file directory
    :param label_map_path: output file
    """
    label_dict = {}
    file_names = os.listdir(train_data_dir)
    file_names.sort()
    for file_name in tqdm(file_names):
        if not '.deft' in file_name:
            print(file_name)
            continue
        with open(os.path.join(train_data_dir, file_name)) as fin:
            for row in fin:
                if not row.strip():
                    continue
                row = row.split()
                assert len(row) == 8
                label_dict[row[4]] = label_dict.get(row[4], 0) + 1

    with open(label_map_path, 'w') as fout:
        for label, freq in sorted(label_dict.items(), key=lambda x: x[1], reverse=True):
            fout.write(f"{label}\t{label_dict[label]}\n")


if __name__ == "__main__":
    train_data_dir = "/home1/sxy/datasets/AdvancedNLP/Data/train_and_dev/train"
    label_map_path = "/home1/sxy/datasets/AdvancedNLP/Data/train_and_dev/label_map.txt"
    generate_label_map(train_data_dir, label_map_path)

