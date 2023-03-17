# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-19 14:19:35
"""
import torch
import torch.nn as nn
import torch.optim as optim


def get_optimizer(model: nn.Module, parameters=[], optim_type="SGD", lr=0.01, split_params=False, **kwargs):
    """
    最终的训练参数是： model.parameters() + parameters
    :param model: nn.Module 模型
    :param parameters: 模型训练参数
    :param optim_type: 优化器, SGD,Adam和AdamW
    :param lr: 学习率
    :param split_params: True: 分离可训练参数，BN层和卷积的bias不加入正则化
                         False: 所以参数，包括BN层和卷积的bias都加入正则化
    :return:
    """
    if split_params:
        params = regularize_parameters(model, weight_decay=kwargs["weight_decay"])
    else:
        params = [{
            'params': list(model.parameters()) + parameters if parameters else list(model.parameters()),
            'weight_decay': kwargs["weight_decay"]
        }]

    if optim_type.lower() == "SGD".lower():
        optimizer = optim.SGD(params, lr=lr, momentum=kwargs["momentum"])
    elif optim_type.lower() == "Adam".lower():
        # β1和β2是加权平均数,用于控制一阶动量和二阶动量
        optimizer = optim.Adam(params, lr=lr)
    elif optim_type.lower() == "AdamW".lower():
        optimizer = optim.AdamW(params, lr=lr)
    else:
        raise Exception("Error:{}".format(optim_type))
    return optimizer


def regularize_parameters(model: nn.Module, weight_decay):
    """
    BatchNorm 层的γ和β不应该添加正则化项,
    卷积层和全连接层的bias也往往不用加正则化项.
    :param model:
    :param optimizer:
    :param weight_decay:
    """
    params_decay = []
    params_no_decay = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(model.parameters())) == len(params_decay) + len(params_no_decay)
    return [{'params': params_no_decay}, {'params': params_decay, 'weight_decay': weight_decay}]
