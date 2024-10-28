# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-08-12 20:27:27
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from ..callbacks.callbacks import Callback
from .WarmUpLR import WarmUpLR
from torch.optim import lr_scheduler


class CosineAnnealingLR(Callback):
    def __init__(self,
                 optimizer,
                 epochs,
                 num_steps,
                 num_cycles=1,
                 lr_init=0.1,
                 decay=0.99,
                 num_warn_up=10,
                 ):
        """
        余弦退火学习率调整策略
        optimizer (Optimizer): Wrapped optimizer.
        t_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.
        :param optimizer:
        :param epochs:
        :param num_steps: 一个epoch的迭代次数，len(self.train_dataloader)
        :param num_cycles: 周期次数
        :param lr_init: is init lr.
        :param decay: 振幅衰减率,当1.0时,每次周期循环不衰减
        :param num_warn_up:
        """
        super(CosineAnnealingLR, self).__init__()
        self.num_steps = num_steps
        self.num_cycles = num_cycles
        self.num_warn_up = num_warn_up
        self.decay = decay
        self.lr_init = lr_init
        self.optimizer = optimizer
        self.epoch = 0
        self.epochs = epochs - self.num_warn_up - 1
        t_max = self.epochs * 1.0 / (2 * self.num_cycles - 1)  # 一次学习率周期的迭代次数，即 T_max 个 epoch 之后重新设置学习率。
        eta_min = 0.00001  # 最小学习率，即在一个周期中，学习率最小会下降到 eta_min，默认值为 0
        self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=eta_min, last_epoch=-1)
        self.warm_up = WarmUpLR(optimizer,
                                num_steps=self.num_steps,
                                lr_init=lr_init,
                                num_warn_up=num_warn_up)

    def get_lr(self, epoch):
        epoch = epoch - self.num_warn_up
        self.scheduler.step(epoch)
        # self.scheduler.step()
        # self.scheduler.last_epoch =epoch
        lr = self.optimizer.param_groups[0]["lr"]
        lr = lr * self.decay ** ((self.num_cycles - 0.5) * epoch / self.epochs)
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
        # total_step = self.num_steps * epoch + step
        # self.scheduler.step(epoch)
        self.warm_up.step(epoch, step)
