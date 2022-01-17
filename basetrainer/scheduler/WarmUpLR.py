# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-08-13 09:13:03
"""
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from ..callbacks.callbacks import Callback
from torch.optim import lr_scheduler


class WarmUpLR(Callback):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self,
                 optimizer,
                 num_steps,
                 lr_init=0.01,
                 num_warn_up=0):
        super(WarmUpLR, self).__init__()
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.lr_init = lr_init
        self.warmup = num_warn_up * self.num_steps
        self.epoch = 0
        super(WarmUpLR, self).__init__()

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self, total_step):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        lr = self.lr_init / self.warmup * total_step
        # lr= self.optimizer.param_groups[0]["lr"]
        return lr

    def on_epoch_begin(self, epoch, logs: dict = {}):
        self.epoch = epoch

    def on_batch_end(self, batch, logs: dict = {}):
        self.step(epoch=self.epoch, step=batch)

    def step(self, epoch=0, step=0):
        total_step = self.num_steps * epoch + step
        if self.warmup > 0 and total_step <= self.warmup:
            lr = self.get_lr(total_step)
            self.set_lr(lr)
