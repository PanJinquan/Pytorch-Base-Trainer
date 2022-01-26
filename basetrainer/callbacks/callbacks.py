# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2021-07-26 10:54:14
# --------------------------------------------------------
"""
from ..engine import base, comm


class Callback(base.Base):
    def __init__(self):
        self.model = None
        self.is_main_process = comm.is_main_process()

    def set_model(self, model):
        self.model = model

    def on_batch_begin(self, batch, logs: dict = {}):
        pass

    def on_batch_end(self, batch, logs: dict = {}):
        pass

    def on_train_summary(self, inputs, outputs, losses, epoch, batch, logs: dict = {}):
        pass

    def on_test_summary(self, inputs, outputs, losses, epoch, batch, logs: dict = {}):
        pass

    def on_epoch_begin(self, epoch, logs: dict = {}):
        pass

    def on_epoch_end(self, epoch, logs: dict = {}):
        pass

    def on_train_begin(self, logs: dict = {}):
        pass

    def on_train_end(self, logs: dict = {}):
        pass

    def on_test_begin(self, logs: dict = {}):
        pass

    def on_test_end(self, logs: dict = {}):
        pass
