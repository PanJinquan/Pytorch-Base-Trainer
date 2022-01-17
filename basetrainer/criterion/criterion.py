# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Face-Recognize-Pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2019-12-31 09:11:25
# --------------------------------------------------------
"""
import torch
import torch.nn as nn


def get_criterion(loss_type: str, num_classes=None, weight=None, device="cuda:0"):
    if loss_type == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif loss_type == "L1Loss":
        criterion = nn.L1Loss(reduction='mean')
    elif loss_type == "mse":
        # loss = nn.MSELoss(reduce=True, size_average=True)
        criterion = nn.MSELoss(reduction='mean')
    else:
        raise Exception("Error:{}".format(loss_type))
    print("loss_type:{}".format(loss_type))
    return criterion
