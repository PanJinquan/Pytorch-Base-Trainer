# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-28 11:32:42
"""
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from .eval_tools.metrics import AverageMeter, accuracy
from ..callbacks import callbacks
from .eval_tools import classification_report
from ..utils import log
from pybaseutils import file_utils



class AccuracyRecorder(callbacks.Callback):
    def __init__(self, target_names=None, indicator="Acc", confusion_matrix=""):
        """
        计算Accuracy的回调函数
        :param target_names: list of str of shape (n_labels,), default=None
                             Optional display names matching the labels (same order).
        :param indicator: 指标名称
        :param confusion_matrix: 计算和保存混淆矩阵的目录
        """
        super(AccuracyRecorder, self).__init__()
        self.target_names = target_names
        self.indicator = indicator
        self.confusion_matrix = confusion_matrix
        self.train_top1 = AverageMeter()
        self.test_top1 = AverageMeter()
        self.true_labels = np.ones(0)
        self.pred_labels = np.ones(0)
        self.logger = log.get_logger()
        self.epoch = 0
        self.test_max_acc = 0

    def on_epoch_begin(self, epoch, logs: dict = {}):
        self.train_top1.reset()
        self.epoch = epoch

    def on_test_begin(self, logs: dict = {}):
        self.test_top1.reset()
        self.true_labels = np.ones(0)
        self.pred_labels = np.ones(0)

    def on_test_end(self, logs: dict = {}):
        acc = accuracy_score(self.true_labels, self.pred_labels) * 100.0
        report = classification_report.get_classification_report(self.true_labels,
                                                                 self.pred_labels,
                                                                 target_names=self.target_names)
        self.logger.info("\nAcc:{:.4f}\n{}".format(acc, report))
        if self.confusion_matrix and acc >= self.test_max_acc:
            self.test_max_acc = acc
            file_utils.remove_prefix_files(self.confusion_matrix, "confusion_matrix_*")
            confuse_file = os.path.join(self.confusion_matrix,
                                        "confusion_matrix_{:0=3d}_{:.4f}.csv".format(self.epoch, acc))
            conf_matrix = classification_report.get_confusion_matrix(self.true_labels,
                                                                     self.pred_labels,
                                                                     self.target_names,
                                                                     filename=confuse_file)
            self.logger.info("save confuse file in:{} ".format(confuse_file))

    @staticmethod
    def summary(phase, average_meter: AverageMeter, indicator, inputs, outputs, logs: dict = {}):
        if isinstance(inputs, dict):
            targets, labels = inputs['image'], inputs["label"].cpu()
        else:
            targets, labels = inputs[0], inputs[1].cpu()
        if isinstance(outputs, tuple): outputs = outputs[0]
        outputs = torch.nn.functional.softmax(outputs, dim=1).cpu()
        pred_score, pred_index = torch.max(outputs, dim=1)
        acc, = accuracy(outputs.data, labels, topk=(1,))
        # fix a bug: n = labels.size(0)
        average_meter.update(acc.data.item(), labels.size(0))
        logs[phase] = logs[phase] if phase in logs else {}
        logs[phase][indicator] = average_meter.avg
        return pred_index, labels

    def on_train_summary(self, inputs, outputs, losses, epoch, step, logs: dict = {}):
        # measure accuracy
        self.summary(phase="train", average_meter=self.train_top1, indicator=self.indicator,
                     inputs=inputs, outputs=outputs, logs=logs)

    def on_test_summary(self, inputs, outputs, losses, epoch, batch, logs: dict = {}):
        # measure accuracy
        pred_index, labels = self.summary(phase="test", average_meter=self.test_top1, indicator=self.indicator,
                                          inputs=inputs, outputs=outputs, logs=logs)
        # get predict result
        self.true_labels = np.hstack([self.true_labels, labels.numpy()])
        self.pred_labels = np.hstack([self.pred_labels, pred_index.numpy()])
