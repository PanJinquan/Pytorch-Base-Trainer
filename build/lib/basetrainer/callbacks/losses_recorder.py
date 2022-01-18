# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-28 11:32:42
"""
import numpy as np
import torch
from ..metric.eval_tools.metrics import AverageMeter
from ..callbacks import callbacks


class LossesRecorder(callbacks.Callback):
    def __init__(self, indicator="loss"):
        """
        record loss
        :param indicator: 指标名称
        """
        super(LossesRecorder, self).__init__()
        self.indicator = indicator
        self.train_losses = AverageMeter()
        self.test_losses = AverageMeter()

    def on_test_begin(self, logs: dict = {}):
        self.train_losses.reset()
        self.test_losses.reset()

    @staticmethod
    def summary(phase, average_meter: AverageMeter, indicator, losses, logs: dict = {}):
        average_meter.update(losses.data.item())
        logs[phase] = logs[phase] if phase in logs else {}
        logs[phase][indicator] = average_meter.avg

    def on_train_summary(self, inputs, outputs, losses, epoch, step, logs: dict = {}):
        self.summary(phase="train",
                     average_meter=self.train_losses,
                     indicator=self.indicator,
                     losses=losses, logs=logs)

    def on_test_summary(self, inputs, outputs, losses, epoch, batch, logs: dict = {}):
        self.summary(phase="test",
                     average_meter=self.test_losses,
                     indicator=self.indicator,
                     losses=losses, logs=logs)
