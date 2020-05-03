# -*- coding: utf-8 -*-
# @Time        : 2020/5/3 22:00
# @Author      : ssxy00
# @File        : linear_schedule_with_warmup.py
# @Description : set lr schedule

class LinearDecayWithWarmup:
    def __init__(self, total_steps, warmup_steps, max_lr, optimizer):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.optimizer = optimizer
        self._step = 0

    def state_dict(self):
        return {'step': self._step,
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step

        if step < self.warmup_steps:
            return self.max_lr * float(step) / float(max(1, self.warmup_steps))
        return self.max_lr * max(0.0, float(self.total_steps - step) / float(max(1, self.total_steps - self.warmup_steps)))