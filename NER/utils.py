# -*- coding: utf-8 -*-
# @Time        : 2020/5/3 21:37
# @Author      : ssxy00
# @File        : utils.py
# @Description :

import random
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def collate_fn(data):
    tensor_tokens, tensor_labels = zip(*data)
    return torch.stack(tensor_tokens), torch.stack(tensor_labels)