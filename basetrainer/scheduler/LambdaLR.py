# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-15 20:06:20
    @Brief  :
"""
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
from ..callbacks.callbacks import Callback
from .WarmUpLR import WarmUpLR
from torch.optim import lr_scheduler


class LambdaLR(Callback):
    def __init__(self,
                 optimizer,
                 epochs,
                 num_steps,
                 linear_lr=False,
                 lr_init=0.1,
                 lrf=0.2,
                 num_warn_up=0,
                 ):
        """
        学习率调整策略,来源于YOLOv5
        :param optimizer:
        :param epochs:
        :param num_steps: 一个epoch的迭代次数，len(self.train_dataloader)
        :param lr_init: is init lr.
        :param num_warn_up:
        """
        super(LambdaLR, self).__init__()
        self.num_warn_up = num_warn_up
        self.num_steps = num_steps
        self.epochs = epochs
        self.epoch = 0
        self.lr_init = lr_init
        self.optimizer = optimizer
        # Scheduler
        if linear_lr:
            lf = lambda x: (1 - x / (self.epochs - 1)) * (1.0 - lrf) + lrf  # linear
        else:
            lf = self.one_cycle(1, lrf, self.epochs)  # cosine 1->hyp['lrf']
        self.scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        # self.scheduler.last_epoch = 0  # do not move
        self.warm_up = WarmUpLR(optimizer,
                                num_steps=self.num_steps,
                                lr_init=lr_init,
                                num_warn_up=num_warn_up)

    @staticmethod
    def one_cycle(y1=0.0, y2=1.0, steps=100):
        # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
        return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def on_epoch_begin(self, epoch, logs: dict = {}):
        self.epoch = epoch - self.num_warn_up
        # self.set_lr(self.get_lr(epoch))
        self.scheduler.step()

    def on_batch_end(self, batch, logs: dict = {}):
        self.step(epoch=self.epoch, step=batch)

    def step(self, epoch=0, step=0):
        # step每次迭代都会调用，比较耗时，建议与step无关的操作放在on_epoch_begin中
        self.warm_up.step(epoch, step)
