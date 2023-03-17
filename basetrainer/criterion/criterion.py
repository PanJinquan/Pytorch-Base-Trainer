# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Face-Recognize-Pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2019-12-31 09:11:25
# --------------------------------------------------------
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Callable


def get_criterion(loss_type: str, num_classes=None, class_weight=None, device="cuda:0", **kwargs):
    """
    :param loss_type: loss_type={loss_name: loss_weigth}
                      FocalLoss,CrossEntropyLoss,LabelSmooth
    :param num_classes:
    :param class_weight: 类别loss权重， a manual rescaling weight given to each class.
                         If given, has to be a Tensor of size `Class`
    :return:
    """
    loss_type = loss_type.lower()
    if loss_type == "CrossEntropyLoss".lower() or loss_type == "CELoss".lower():
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    elif loss_type == "L1Loss".lower():
        criterion = nn.L1Loss(reduction='mean')
    elif loss_type == "mse".lower():
        # loss = nn.MSELoss(reduce=True, size_average=True)
        criterion = nn.MSELoss(reduction='mean')
    else:
        raise Exception("Error:{}".format(loss_type))
    print("loss_type:{}".format(loss_type))
    return criterion


def build_criterion(loss_type: str or List[str] or Dict[str, float],
                    num_classes=None,
                    class_weight=None,
                    device="cuda:0",
                    **kwargs):
    """
    使用nn.BCELoss需要在该层前面加上Sigmoid函数
    使用nn.CrossEntropyLoss会自动加上Softmax层,所以最后一层不需要加上Softmax()
    :param loss_type: loss_type={loss_name: loss_weigth}
                      FocalLoss,CrossEntropyLoss,LabelSmooth
    :param num_classes:
    :param class_weight: 类别loss权重， a manual rescaling weight given to each class.
                         If given, has to be a Tensor of size `Class`
    :return:
    """
    # logger = log.get_logger()
    if isinstance(class_weight, np.ndarray):
        class_weight = torch.from_numpy(class_weight.astype(np.float32)).to(device)
    if isinstance(loss_type, str):
        loss_type = [loss_type]
    if isinstance(loss_type, list):
        loss_type = {loss: 1.0 for loss in loss_type}
    assert isinstance(loss_type, dict)
    criterions = {}
    weights = {}
    for loss, loss_weight in loss_type.items():
        criterion = get_criterion(loss, num_classes, class_weight, device=device, **kwargs)
        criterions[loss] = criterion
        weights[loss] = loss_weight
    criterions = ComposeLoss(criterions=criterions, weights=weights)
    # logger.info("use criterions:{}".format(weights))
    return criterions


class ComposeLoss(nn.Module):
    def __init__(self, criterions: Dict[str, Callable], weights: Dict[str, float] = None):
        """
        联合LOSS函数
        :param criterions: Dict[str, Callable], Loss函数,
                         ==> {"Loss1": Loss1Function,"Loss2":Loss2Function}
        :param  weights: Dict[str, float] Loss的权重
                         ==> {"Loss1": 1.0,"Loss2":1.0}
        """
        super(ComposeLoss, self).__init__()
        if isinstance(weights, dict):
            assert criterions.keys() == weights.keys(), \
                Exception("Key Error:criterions:{},weights:{}".format(criterions.keys(), weights.keys()))
        self.weights = weights
        self.criterions = criterions

    def parameters(self, recurse: bool = True):
        param = []
        for loss in self.criterions.values():
            param += loss.parameters()
        return param

    def __call__(self, logits, labels):
        losses = {}
        for name, criterion in self.criterions.items():
            loss = criterion(logits, labels)
            if isinstance(loss, dict):
                losses = {k: v * self.weights[name] for k, v in loss.items()} if self.weights else loss
            else:
                losses[name] = loss * self.weights[name] if self.weights else loss
        return losses


if __name__ == "__main__":
    from basetrainer.utils import torch_tools

    torch_tools.set_env_random_seed()
    batch_size = 16
    num_classes = 10
    input = torch.ones(batch_size, num_classes, 10, 10).cuda()
    target = torch.ones(batch_size, 10, 10).long().cuda()
    loss_type = {"CELoss": 0.5, "CELoss": 0.5}
    losses = build_criterion(loss_type, num_classes=num_classes, device="cuda:0")
    loss = losses(input, target)
    print(loss)
