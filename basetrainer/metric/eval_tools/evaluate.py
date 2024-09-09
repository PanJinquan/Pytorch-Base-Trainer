# -*-coding: utf-8 -*-
"""
    @Project: torch-Face-Recognize-Pipeline
    @File   : evaluate.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-12-26 16:32:47
"""

import torch

from basetrainer.metric.eval_tools.metrics import AverageMeter, accuracy


class EvaluationMetrics(object):
    def __init__(self, model=None, device="cuda:0"):
        self.model = model
        self.device = device

    def update_model(self, model, gpu_id):
        """
        :param model:
        :param gpu_id:
        :return:
        """
        if len(gpu_id) > 1:
            model = model.module
            self.model = model.to(self.device)
        else:
            self.model = model.to(self.device)
        # switch to evaluation mode
        self.model.eval()

    def forward(self, input_tensor):
        """
        :param input_tensor: input tensor
        :return:
        """
        with torch.no_grad():
            out_tensor = self.model(input_tensor.to(self.device))
        return out_tensor


    def evaluation(self,inputs,labels, epoch):
        """
        val data metrics
        :param epoch:
        :return:
        """
        self.model.eval()  # set to training mode
        losses = AverageMeter()
        top1 = AverageMeter()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)
        # measure accuracy and record loss
        acc, = accuracy(outputs.data, labels, topk=(1,))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(acc.data.item(), inputs.size(0))
