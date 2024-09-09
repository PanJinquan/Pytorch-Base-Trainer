# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-28 15:32:44
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from .WarmUpLR import WarmUpLR
from ..callbacks.callbacks import Callback


class ExponentialLR(Callback):
    def __init__(self,
                 optimizer,
                 epochs,
                 num_steps,
                 lr_init=0.01,
                 num_warn_up=0,
                 decay=0.90):
        """
        指数衰减学习率
        :param optimizer:
        :param epochs:
        :param num_steps: 一个epoch的迭代次数，len(self.train_dataloader)
        :param milestones:  (list): List of epoch indices. Must be increasing.
        :param lr_init: is init lr.
        :param num_warn_up:
        :param gamma (float): 学习率衰减率learning rate decay.Default: 0.9
        """
        self.optimizer = optimizer
        self.num_warn_up = num_warn_up
        self.epochs = epochs - self.num_warn_up
        self.num_steps = num_steps
        self.max_step = self.epochs * self.num_steps
        self.lr_init = lr_init
        self.epoch = 0
        self.decay = decay
        self.warm_up = WarmUpLR(optimizer,
                                num_steps=self.num_steps,
                                lr_init=lr_init,
                                num_warn_up=num_warn_up)
        super(ExponentialLR, self).__init__()

    def get_lr(self, epoch):
        epoch = epoch - self.num_warn_up
        # lr = self.optimizer.param_groups[0]["lr"]
        # lr = self.lr_init * self.gamma ** epoch
        lr = self.lr_init * self.decay ** (50 * epoch / self.epochs)
        return lr

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def on_epoch_begin(self, epoch, logs: dict = {}):
        self.epoch = epoch
        self.set_lr(self.get_lr(epoch))

    def on_batch_end(self, batch, logs: dict = {}):
        self.step(epoch=self.epoch, step=batch)

    def step(self, epoch=0, step=0):
        # step每次迭代都会调用，比较耗时，建议与step无关的操作放在on_epoch_begin中
        self.warm_up.step(epoch, step)
