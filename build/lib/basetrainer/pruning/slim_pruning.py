# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-12-10 17:39:21
"""
import os
import torch
import torch.nn as nn
import torch_pruning as pruning


def model_pruning(model: nn.Module,
                  input_size=[1, 3, 128, 128],
                  sparsity=0.2,
                  prune_mod="fpgm",
                  prune_shortcut=False,
                  device="cpu",
                  **kwargs):
    """
    使用Slim-pruner进行模型剪枝和压缩
    https://github.com/yanggui19891007/Pytorch-Auto-Slim-Tools
    :param model:  Pytorch模型
    :param input_size: 模型输入的维度[batch-size,channel,H,W]
    :param sparsity: 模型稀疏力度,值越大,模型压缩越大，计算量越小；
    :param device:
    :return:
    """
    input_res = tuple(input_size[1:])
    model.cpu()
    dummy_input = torch.randn(input_size).cpu()
    flops_raw, params_raw = pruning.get_model_complexity_info(model,
                                                              input_res=input_res,
                                                              as_strings=True,
                                                              print_per_layer_stat=False)
    print('-[INFO] before pruning flops:  ' + flops_raw)
    print('-[INFO] before pruning params:  ' + params_raw)
    # 剪枝引擎建立
    slim = pruning.Autoslim(model, inputs=dummy_input, compression_ratio=sparsity)
    if prune_mod == 'fpgm':
        config = {
            'layer_compression_ratio': None,
            'norm_rate': 1.0, 'prune_shortcut': 1,
            'dist_type': 'l1', 'pruning_func': 'fpgm'
        }
    elif prune_mod == 'l1':
        config = {
            'layer_compression_ratio': None,
            'norm_rate': 1.0, 'prune_shortcut': 1,
            'global_pruning': False, 'pruning_func': 'l1'
        }

    slim.base_prunging(config)
    flops_new, params_new = pruning.get_model_complexity_info(model,
                                                              input_res,
                                                              as_strings=True,
                                                              print_per_layer_stat=False)
    print('\n-[INFO] after pruning flops:  ' + flops_new)
    print('-[INFO] after pruning params:  ' + params_new)
    model = model.to(device)
    return model


if __name__ == "__main__":
    from basetrainer.utils import torch_tools
    from torchvision.models.resnet import resnet50, resnet18

    model = resnet18(pretrained=True)
    torch.save(model.state_dict(), "origin.pth")
    model = model_pruning(model, sparsity=0.2)
    torch.save(model.state_dict(), "pruning.pth")
