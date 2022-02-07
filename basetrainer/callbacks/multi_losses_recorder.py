# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-28 11:32:42
"""
import numpy as np
from typing import Dict, List
from ..metric.eval_tools.metrics import AverageMeter
from . import callbacks


class MultiLossesRecorder(callbacks.Callback):
    def __init__(self, indicators: Dict[str, List] = None):
        """
        用于记录多个loss的值,并自动计算total_loss
        :param indicators: 指标名称(dict), 若indicators = None则表示记录所有loss
               如indicators = {"loss": ["loss1", "loss2"]}，其中"loss1", "loss2"是run_step()中返回的loss值:
               Examples:
                   > outputs = model(inputs)
                   > loss1,loss2=criterion(outputs,targets)
                   > losses = {"loss1": loss1, "loss2": loss2}
                   > return outputs, losses
        """
        super(MultiLossesRecorder, self).__init__()
        self.indicators = indicators
        self.train_losses = {}
        self.test_losses = {}

    def init_train_recorder(self):
        if self.train_losses:
            return
        for name, indicator in self.indicators.items():
            for k in indicator:
                self.train_losses[k] = AverageMeter()

    def init_test_recorder(self):
        if self.test_losses:
            return
        for name, indicator in self.indicators.items():
            for k in indicator:
                self.test_losses[k] = AverageMeter()

    def init_indicators(self, losses: dict = {}):
        if self.indicators:
            return
        self.indicators = {"loss": list(losses.keys())}

    def on_epoch_end(self, epoch, logs: dict = {}):
        for name, indicator in self.indicators.items():
            for k in indicator:
                if k in self.train_losses:
                    self.train_losses[k].reset()
                if k in self.test_losses:
                    self.test_losses[k].reset()

    @staticmethod
    def summary(phase, average_meter: dict, indicators: dict, losses, logs: dict = {}):
        logs[phase] = logs[phase] if phase in logs else {}
        for name, indicator in indicators.items():
            scalar_dict = {}
            for k in indicator:
                average_meter[k].update(losses[k].data.item())
                # average_meter[k].update(losses[k].data.item(), labels.size(0))
                scalar_dict[k] = average_meter[k].avg
            if len(indicator) > 1:
                scalar_dict["total_{}".format(name)] = sum(scalar_dict.values())
            logs[phase][name] = scalar_dict

    def on_train_summary(self, inputs, outputs, losses, epoch, step, logs: dict = {}):
        self.init_indicators(losses)
        self.init_train_recorder()
        self.summary(phase="train", average_meter=self.train_losses,
                     indicators=self.indicators, losses=losses, logs=logs)

    def on_test_summary(self, inputs, outputs, losses, epoch, batch, logs: dict = {}):
        self.init_indicators(losses)
        self.init_test_recorder()
        self.summary(phase="test", average_meter=self.test_losses,
                     indicators=self.indicators, losses=losses, logs=logs)
