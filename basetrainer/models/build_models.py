# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-01-17 17:46:38
"""

import os
import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict


def get_models(net_type, input_size, num_classes, width_mult=1.0, is_train=True, pretrained=True, **kwargs):
    """
    :param net_type:  resnet18,resnet34,resnet50, mobilenet_v2
    :param input_size: 模型输入大小
    :param num_classes: 类别数
    :param width_mult:
    :param is_train:
    :param pretrained:
    :param kwargs:
    :return:
    """
    if net_type.lower().startswith("resnet"):
        model = resnet_model(net_type,
                             num_classes=num_classes,
                             pretrained=pretrained)
    elif net_type.lower() == "mobilenet_v2":
        model = mobilenet_v2(num_classes=num_classes,
                             width_mult=width_mult,
                             pretrained=pretrained)
    else:
        raise Exception("Error: net_type:{}".format(net_type))
    return model


def resnet_model(net_type, num_classes, pretrained=True):
    """
    :param net_type: resnet18,resnet34,resnet50
    :param num_classes: if None ,return no-classifier-layers backbone
    :param pretrained: <bool> pretrained
    :return:
    """
    if net_type.lower() == "resnet18":
        backbone = models.resnet18(pretrained=pretrained)
        out_channels = 512
        expansion = 1
    elif net_type.lower() == "resnet34":
        backbone = models.resnet34(pretrained=pretrained)
        out_channels = 512
        expansion = 1
    elif net_type.lower() == "resnet50":
        backbone = models.resnet50(pretrained=pretrained)
        out_channels = 512
        expansion = 4
    else:
        raise Exception("Error: net_type:{}".format(net_type))

    if num_classes:
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        assert backbone.fc.in_features == out_channels * expansion
        backbone.fc = nn.Linear(out_channels * expansion, num_classes)
    else:
        # remove mobilenet_v2  classifier layers
        model_dict = OrderedDict(backbone.named_children())
        model_dict.pop("avgpool")
        model_dict.pop("fc")
        backbone = torch.nn.Sequential(model_dict)
        # if attention:
        #     backbone.add_module("attention", ChannelAttention(input_size=last_channel))
    return backbone


def mobilenet_v2(num_classes=None, width_mult=1.0, pretrained=False):
    """
    :param pretrained: <bool> pretrained
    :param num_classes: if None ,return no-classifier-layers backbone
    :param last_channel:
    :param width_mult:
    :return:
    """
    model = models.mobilenet_v2(pretrained=pretrained, width_mult=width_mult)
    # state_dict1 = model.state_dict()
    if num_classes:
        last_channel = model.last_channel
        # replace mobilenet_v2  classifier layers
        classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )
        model.classifier = classifier
    else:
        # remove mobilenet_v2  classifier layers
        model_dict = OrderedDict(model.named_children())
        model_dict.pop("classifier")
        model = torch.nn.Sequential(model_dict)
        # state_dict2 = model.state_dict()
    return model


if __name__ == "__main__":
    from basetrainer.utils import torch_tools
    net_type = "resnet18"
    net_type = "mobilenet_v2"
    input_size = [320, 320]
    model = get_models(net_type=net_type, input_size=input_size, num_classes=10, pretrained=False)
    torch_tools.nni_summary_model(model, batch_size=1, input_size=input_size, plot=False)
