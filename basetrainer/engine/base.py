# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2021-07-26 10:54:14
# --------------------------------------------------------
"""


class Base(object):
    def __init__(self):
        self.model = None

    def set_model(self, model):
        """设置模型"""
        self.model = model

    def on_batch_begin(self, batch, logs: dict = {}):
        """每次迭代开始时回调"""
        pass

    def on_batch_end(self, batch, logs: dict = {}):
        """每次迭代结束时回调"""
        pass

    def on_train_summary(self, inputs, outputs, losses, epoch, step, logs: dict = {}):
        """每次迭代，训练结束时回调"""
        pass

    def on_test_summary(self, inputs, outputs, losses, epoch, step, logs: dict = {}):
        """每次迭代，测试结束时回调"""
        pass

    def on_epoch_begin(self, epoch, logs: dict = {}):
        """每个epoch开始时回调"""
        pass

    def on_epoch_end(self, epoch, logs: dict = {}):
        """每个epoch结束时回调"""
        pass

    def on_train_begin(self, logs: dict = {}):
        """开始训练时回调"""
        pass

    def on_train_end(self, logs: dict = {}):
        """结束训练时回调"""
        pass

    def on_test_begin(self, logs: dict = {}):
        """开始测试时回调"""
        pass

    def on_test_end(self, logs: dict = {}):
        """结束测试时回调"""
        pass
