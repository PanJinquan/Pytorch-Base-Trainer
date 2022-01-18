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


class MultiStepLR(Callback):
    def __init__(self,
                 optimizer,
                 epochs,
                 num_steps,
                 milestones,
                 lr_init=0.01,
                 num_warn_up=0,
                 gamma=0.1):
        """
        a cosine decay scheduler about steps, not epochs.
        :param optimizer: ex. optim.SGD
        :param epochs:
        :param num_steps: 一个epoch的迭代次数，len(self.train_dataloader)
        :param milestones:  (list): List of epoch indices. Must be increasing.
        :param lr_init: lr_max is init lr.
        :param num_warn_up:
        :param gamma (float): Multiplicative factor of learning rate decay.Default: 0.1.
        """
        self.optimizer = optimizer
        self.epochs = epochs
        self.num_steps = num_steps
        self.max_step = epochs * self.num_steps
        self.lr_init = lr_init
        self.milestones = milestones
        self.milestones.sort()
        self.lr_list = [lr_init * gamma ** decay for decay in range(0, len(self.milestones) + 1)]
        self.epoch = 0
        self.warm_up = WarmUpLR(optimizer,
                                num_steps=self.num_steps,
                                lr_init=lr_init,
                                num_warn_up=num_warn_up)
        super(MultiStepLR, self).__init__()

    def get_lr(self, epoch, milestones, lr_list):
        lr = self.optimizer.param_groups[0]["lr"]
        max_stages = milestones[-1]
        if epoch < max_stages:
            for index in range(len(milestones)):
                if epoch < milestones[index]:
                    lr = lr_list[index]
                    break
        elif epoch >= max_stages:
            lr = lr_list[-1]
        return lr

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def set_milestones_lr(self, epoch, milestones, lr_list):
        '''
        :param epoch:
        :param milestones: [    35, 65, 95, 150]
        :param lr_list:   [0.1, 0.01, 0.001, 0.0001, 0.00001]
        :return:
        '''
        lr = self.get_lr(epoch, milestones, lr_list)
        self.set_lr(lr)

    def on_epoch_begin(self, epoch, logs: dict = {}):
        self.epoch = epoch
        self.set_milestones_lr(epoch, self.milestones, self.lr_list)

    def on_batch_end(self, batch, logs: dict = {}):
        self.step(epoch=self.epoch, step=batch)

    def step(self, epoch=0, step=0):
        # total_step = self.num_steps * epoch + step
        # step每次迭代都会调用，比较耗时，建议与step无关的操作放在on_epoch_begin中
        # self.set_milestones_lr(epoch, self.milestones, self.lr_list)
        self.warm_up.step(epoch, step)
