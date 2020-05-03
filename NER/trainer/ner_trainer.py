# -*- coding: utf-8 -*-
# @Time        : 2020/5/3 17:19
# @Author      : ssxy00
# @File        : ner_trainer.py
# @Description :

import os
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from trainer.linear_schedule_with_warmup import LinearDecayWithWarmup
from utils import collate_fn


class NerTrainer:
    def __init__(self, model, args, train_dataset, dev_dataset, tokenizer, label2idx):
        # dev_data_loader = DataLoader(dev_dataset, batch_size=4, collate_fn=collate_fn)
        # for tensor_tokens, tensor_labels in dev_data_loader:
        #     print(tensor_tokens)
        self.device = torch.device(args.device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset

        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size

        # checkpoint
        self.model_dir = args.model_dir
        self.save_interval = args.save_interval

        # log
        self.writer = SummaryWriter(args.log_dir)

        # criterion
        self.label_pad_id = self.label2idx["PAD"]
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.label_pad_id)

        # optimizer
        self.clip_grad = args.clip_grad
        lr = args.lr
        base_optimizer = AdamW(self.model.parameters(), lr=lr, correct_bias=True)
        total_steps = math.ceil(self.n_epochs * len(self.train_dataset) / self.batch_size)
        warmup_steps = int(args.warmup_proportion * total_steps)
        self.optimizer = LinearDecayWithWarmup(total_steps, warmup_steps, lr, base_optimizer)

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _eval_train(self, epoch):
        self.model.train()
        ave_loss = 0
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=collate_fn, num_workers=4)

        tqdm_data = tqdm(train_dataloader, desc='Train (epoch #{})'.format(epoch))

        for i, (tensor_tokens, tensor_labels) in enumerate(tqdm_data):
            tensor_tokens = tensor_tokens.to(self.device)
            tensor_labels = tensor_labels.to(self.device)
            attention_mask = tensor_tokens.ne(self.tokenizer.pad_token_id).float()
            logits = self.model(input_ids=tensor_tokens, attention_mask=attention_mask)
            loss = self.criterion(logits.view(-1, logits.shape[-1]), tensor_labels.view(-1))

            loss.backward()
            if self.clip_grad is not None:
                for group in self.optimizer.param_groups:
                    nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

                self.optimizer.step()
                self.optimizer.zero_grad()

            ave_loss = (ave_loss * i + loss.item()) / (i + 1)

            tqdm_data.set_postfix({'lr': self.optimizer.param_groups[0]['lr'],
                                   'loss': loss.item(),
                                   'ave_loss': ave_loss,
                                   })
            self.writer.add_scalar("Train/loss", loss.item(), epoch * len(tqdm_data) + i)
            self.writer.add_scalar("Train/average loss", ave_loss, epoch * len(tqdm_data) + i)
            self.writer.add_scalar("Train/lr", self.optimizer.param_groups[0]['lr'], epoch * len(tqdm_data) + i)

    def _eval_dev(self, epoch):
        self.model.eval()
        ave_loss = 0
        n_correct = 0
        n_golden = 0
        n_correct_without_O = 0
        n_golden_without_O = 1e-5  # 防止除 0
        dev_dataloader = DataLoader(self.dev_dataset, batch_size=8, shuffle=False,
                                    collate_fn=collate_fn, num_workers=4)

        tqdm_data = tqdm(dev_dataloader, desc='Dev (epoch #{})'.format(epoch))

        with torch.no_grad():
            for i, (tensor_tokens, tensor_labels) in enumerate(tqdm_data):
                tensor_tokens = tensor_tokens.to(self.device)
                tensor_labels = tensor_labels.to(self.device)
                attention_mask = tensor_tokens.ne(self.tokenizer.pad_token_id).float()
                logits = self.model(input_ids=tensor_tokens, attention_mask=attention_mask)
                predict_labels = logits.argmax(-1)
                # acc
                n_golden += torch.sum(tensor_labels.ne(self.label_pad_id)).item()
                n_correct += torch.sum((predict_labels == tensor_labels) & (predict_labels.ne(0))).item()
                # acc without "O"
                n_golden_without_O += torch.sum(
                    tensor_labels.ne(self.label_pad_id) & tensor_labels.ne(self.label2idx["O"])).item()
                n_correct_without_O += torch.sum(
                    (predict_labels == tensor_labels) & tensor_labels.ne(self.label2idx["O"]) &
                    (predict_labels.ne(0))).item()

                loss = self.criterion(logits.view(-1, logits.shape[-1]), tensor_labels.view(-1))
                ave_loss = (ave_loss * i + loss.item()) / (i + 1)

                tqdm_data.set_postfix({'acc': n_correct / n_golden,
                                       'acc_without_O': n_correct_without_O / n_golden_without_O,
                                       'loss': loss.item(),
                                       'ave_loss': ave_loss,
                                       })
                self.writer.add_scalar("Dev/loss", loss.item(), epoch * len(tqdm_data) + i)
                self.writer.add_scalar("Dev/average loss", ave_loss, epoch * len(tqdm_data) + i)
                self.writer.add_scalar("Dev/acc", n_correct / n_golden, epoch * len(tqdm_data) + i)
                self.writer.add_scalar("Dev/acc without O", n_correct_without_O / n_golden_without_O, epoch * len(tqdm_data) + i)

    def train(self, last_epoch=0):
        print('begin to train')
        for epoch_idx in range(last_epoch + 1, self.n_epochs + 1):
            self._eval_train(epoch_idx)
            self._eval_dev(epoch_idx)
            if epoch_idx % self.save_interval == 0:
                save_dir = os.path.join(self.model_dir, f"checkpoint{epoch_idx}.pt")
                torch.save(self.state_dict(), save_dir)
