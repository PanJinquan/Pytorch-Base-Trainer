# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-12-09 19:16:19
"""
import os
import torch
import torch.nn as nn
import copy
import torch.onnx
from nni.compression.pytorch.utils.counter import count_flops_params
import torch.nn.utils.prune as prune
import torch.nn.functional as F


def model_pruning(model: nn.Module,
                  input_size=[1, 3, 128, 128],
                  sparsity=0.2,
                  prune_mod="L1",
                  reuse=False,
                  speedup=True,
                  output_prune="pruning_output",
                  mask_file="",
                  dependency_aware=True,
                  device="cpu",
                  verbose=False,
                  **kwargs):
    """
    使用torch.nn.utils.prune进行模型剪枝和压缩
    https://pytorch.org/tutorials/intermediate/pruning_tutorial.html?highlight=prune
    :param model:  Pytorch模型
    :param input_size: 模型输入的维度[batch-size,channel,H,W]
    :param sparsity: 模型剪枝目标稀疏度,值越大,模型越稀疏，计算量越小；0.5表示剪除50%
    :param reuse: 是否复现模型剪枝的结果
                False: 进行模型剪枝(pruning+ModelSpeedup),会生成mask文件(mask_naive_l1filter.pth)
                True : 复现模型剪枝的结果,需要提供mask_file文件
    :param speedup: 是否加速模型
    :param output_prune: 模型剪枝输出文件
    :param mask_file: reuse=True需要提供模型剪枝的mask文件,默认保存在output_prune目录下(mask_naive_l1filter.pth)
    :param dependency_aware 依赖感知模式 https://nni.readthedocs.io/zh/stable/Compression/DependencyAware.html
    :param device:
    :return:
    """
    info = ""
    model = model.to(device)
    if not os.path.exists(output_prune): os.makedirs(output_prune)
    prune_file = os.path.join(output_prune, 'pruned_naive_{}filter.pth'.format(prune_mod))
    onnx_file = os.path.join(output_prune, 'pruned_naive_{}filter.onnx'.format(prune_mod))
    mask_file = os.path.join(output_prune, 'mask_naive_{}filter.pth'.format(prune_mod)) if not mask_file else mask_file
    dummy_input = torch.randn(input_size).to(device)
    # 原始模型的计算量和参数量
    flops, params, _ = count_flops_params(model, dummy_input, verbose=verbose)
    info += f"origin-Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M\n"
    # 定义需要剪枝的参数
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # parameters += [(module, 'weight'), (module, 'bias')]
            parameters_to_prune += [(module, 'weight')]
            if hasattr(module, 'bias') and module.bias is not None:
                parameters_to_prune += [(module, 'bias')]
        if isinstance(module, torch.nn.Linear):
            # parameters += [(module, 'weight'), (module, 'bias')]
            parameters_to_prune += [(module, 'weight')]
            if hasattr(module, 'bias') and module.bias is not None:
                parameters_to_prune += [(module, 'bias')]
    # 全局剪枝
    prune.global_unstructured(parameters_to_prune,
                              pruning_method=prune.L1Unstructured,
                              amount=sparsity, )
    # to verify that all masks exist
    # print(dict(model.named_buffers()).keys())
    # apply_compression_results_v1(prune, model)
    # 剪枝模型的计算量和参数量
    flops, params, _ = count_flops_params(model, dummy_input, verbose=verbose)
    info += f"prune-Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M\n"
    print(info)
    return model


def apply_compression_results_v1(prune: torch.nn.utils.prune, model: nn.Module):
    # Apply mask to model tensor and remove mask from state_dict
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
            if hasattr(module, 'bias') and module.bias is not None:
                prune.remove(module, 'bias')
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')
            if hasattr(module, 'bias') and module.bias is not None:
                prune.remove(module, 'bias')


def apply_compression_results_v2(model):
    """
    Apply the masks from ```masks_file``` to the model
    Note: this API is for inference, because it simply multiplies weights with
    corresponding masks when this API is called.
    model : torch.nn.ModuleThe model to be compressed
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.weight.data = module.weight.data.mul_(module.weight_mask)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.mul_(module.bias_mask)
        if isinstance(module, torch.nn.Linear):
            module.weight.data = module.weight.data.mul_(module.weight_mask)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.mul_(module.bias_mask)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc = nn.Linear(128, 256)
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from torchvision.models.resnet import resnet50, resnet18
    from torchvision.models.squeezenet import SqueezeNet
    from torchvision.models.mobilenet import MobileNetV2
    from segment.models import build_model
    from libs.performance import performance

    device = "cuda:0"
    batch_size = 2
    width_mult = 1.0
    num_classes = 20
    input_size = [1, 3, 224, 224]
    net_type = 'modnet_v2'
    model = resnet18(pretrained=True)
    # model = SimpleModel()
    # model = MobileNetV2()
    # model = build_model.get_models(net_type, input_size, num_classes, width_mult=width_mult, pretrained=False)
    model.eval()
    inputs = torch.randn(input_size)
    model = model.to((device))
    inputs = inputs.to((device))
    out = model(inputs)
    performance.model_performance(model, inputs)
    prune_model = copy.deepcopy(model)
    prune_model = model_pruning(prune_model, input_size=input_size, sparsity=0.9, dependency_aware=True, device=device)
    performance.model_performance(model, inputs)
    performance.model_performance(prune_model, inputs)
