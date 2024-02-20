# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-12-09 19:16:19
"""
import os
import copy
import torch
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.algorithms.compression.pytorch import pruning
from nni.compression.pytorch import apply_compression_results


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
    使用NNI进行模型剪枝和压缩
    https://github.com/microsoft/nni/blob/master/docs/en_US/Compression/compression_pipeline_example.ipynb
    use l1filter pruner to prune the model
    Note that if you use a compressor that need you to pass a optimizer,
    you need a new optimizer instead of you have used above, because NNI might modify the optimizer.
    And of course this modified optimizer can not be used in finetuning.
    Usage:
        model = build_model()
        model = model_pruning(model,input_size,sparsity=0.2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for epoch in range(0, epochs):
            trainer(model, optimizer, criterion, epoch)
            evaluator(model)
            torch.save(model.state_dict(), "model_pruning.pth")
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
    :param config: 模型剪枝配置,用于指定需要剪枝网络层；
                        如果不指定op_names,默认对所以层进行剪枝
                        config_list = [{'sparsity': sparsity,'op_types': ['Conv2d'],'op_names': ['conv1']}]
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
    # 模型剪枝,会生成mask文件(mask_naive_l1filter.pth)
    if not reuse:
        """
        Choose a pruner and pruning 
        use l1filter pruner to prune the model
        Note that if you use a compressor that need you to pass a optimizer,
        you need a new optimizer instead of you have used above, because NNI might modify the optimizer.
        And of course this modified optimizer can not be used in finetuning.
        """
        if prune_mod.lower() == "Level".lower():
            config = [{'sparsity': sparsity, 'op_types': ['Conv2d']}]
            pruner = pruning.LevelPruner(model, config)
        elif prune_mod.lower() == "L1".lower():
            # op_types : Only Conv2d is supported in L1FilterPruner.
            # config = [{'sparsity': sparsity, 'op_types': ['Conv2d'], "exclude": False}]
            config = [{'sparsity': sparsity, 'op_types': ['Conv2d']}]
            pruner = pruning.L1FilterPruner(model, config, dependency_aware, dummy_input=dummy_input)
        elif prune_mod.lower() == "L2".lower():
            # op_types : Only Conv2d is supported in L2FilterPruner.
            config = [{'sparsity': sparsity, 'op_types': ['Conv2d']}]
            pruner = pruning.L2FilterPruner(model, config, dependency_aware, dummy_input=dummy_input)
        elif prune_mod.lower() == "FPGM".lower():
            # op_types : Only Conv2d is supported in FPGM Pruner
            config = [{'sparsity': sparsity, 'op_types': ['Conv2d']}]
            pruner = pruning.FPGMPruner(model, config, dependency_aware, dummy_input=dummy_input)
        elif prune_mod.lower() == "Slim".lower():
            # op_types : Only BatchNorm2d is supported in Slim Pruner.
            config = [{'sparsity': sparsity, 'op_types': ['BatchNorm2d']}]
            pruner = pruning.SlimPruner(model,
                                        config,
                                        optimizer=None,
                                        trainer=None,
                                        criterion=None,
                                        sparsifying_training_epochs=10)
        elif prune_mod.lower() == "Slim".lower():
            config = [{'sparsity': sparsity, 'op_types': ['BatchNorm2d']}]
            pruner = pruning.ActivationMeanRankFilterPruner()

        else:
            raise Exception("Error prune_mod:{}".format(prune_mod))
        # compress the model, the mask will be updated.
        pruner.compress()
        # pruner.get_pruned_weights()
        # use a dummy input to apply the sparsify.
        out = model(dummy_input)
        # 剪枝后模型的计算量和参数量
        flops, params, _ = count_flops_params(model, dummy_input, verbose=verbose)
        info += f"pruner-Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M\n"
        # export the sparsified and mask model
        pruner.export_model(model_path=prune_file, mask_path=mask_file,
                            onnx_path=onnx_file, input_shape=dummy_input.shape,
                            device=device,
                            opset_version=11)
        # speedup the model with provided weight mask.If you use a wrapped model, don't forget to unwrap it.
        pruner._unwrap_model()
        # 将掩码应用到模型,模型会变得更小，推理延迟也会减小
        # apply_compression_results(model, mask_file, device)
    if speedup:
        if not os.path.exists(mask_file): raise Exception("not found mask file:{}".format(mask_file))
        print("load mask file to speed up:{}".format(mask_file))
        speed_up = ModelSpeedup(model, dummy_input=dummy_input, masks_file=mask_file)
        speed_up.speedup_model()
        out = model(dummy_input)
        # speedup后模型的计算量和参数量
        flops, params, _ = count_flops_params(model, dummy_input, verbose=verbose)
        info += f"speedup-Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M\n"
    # finetune the model to recover the accuracy.
    model = model.to(device)
    print(info)
    return model


def nni_model_pruning_test(model, input_size=[1, 3, 416, 416], sparsity=0.5, output="output", device="cpu"):
    """
    https://zhuanlan.zhihu.com/p/382638682
    :param model:
    :param input_size:
    :param sparsity:
    :param output:
    :param device:
    :return:
    """
    from ptflops.flops_counter import get_model_complexity_info

    if not os.path.exists(output):
        os.makedirs(output)
    config = [{
        'sparsity': sparsity,
        'op_types': ['Conv2d']
        # 'op_types': ['BatchNorm2d']
    }]
    dummy_input = torch.randn(input_size).to(device)
    #
    origin_path = os.path.join(output, 'origin.pth')
    onnx_path = os.path.join(output, 'origin.onnx')
    torch.save(model.state_dict(), origin_path)
    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                      do_constant_folding=True,
                      verbose=False,
                      export_params=True,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'])

    tmp_model = copy.deepcopy(model).to(device)
    out_tensor = tmp_model(dummy_input)
    flops, params, _ = count_flops_params(tmp_model, dummy_input, verbose=True)
    print(f"Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")

    pruner = pruning.FPGMPruner(tmp_model, config)
    pruner.compress()
    flops, params, _ = count_flops_params(tmp_model, dummy_input, verbose=True)
    print(f"Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")

    pruned_model_path = os.path.join(output, 'slim_pruned.pth')
    pruned_model_mask = os.path.join(output, 'slim_pruned_mask.pth')
    pruned_model_onnx = os.path.join(output, 'slim_pruned.onnx')

    pruner.export_model(model_path=pruned_model_path, mask_path=pruned_model_mask,
                        onnx_path=pruned_model_onnx, input_shape=dummy_input.shape,
                        device=device,
                        opset_version=11)
    tmp_model = copy.deepcopy(model).to(device)
    print('model pruned done.')
    # apply_compression_results
    apply_compression_results(tmp_model, masks_file=pruned_model_mask)
    print('apply_compression_results')
    out_tensor = tmp_model(dummy_input)
    # Speedup
    m_speedup = ModelSpeedup(tmp_model, dummy_input, masks_file=pruned_model_mask)
    m_speedup.speedup_model()
    out_tensor = tmp_model(dummy_input)
    print('speedup_model ')
    # # print(model)
    slim_speedup_path = os.path.join(output, 'slim_speedup_model.pth')
    slim_speedup_onnx = os.path.join(output, 'slim_speedup_model.onnx')
    torch.save(tmp_model.state_dict(), slim_speedup_path)
    torch.onnx.export(tmp_model, dummy_input, slim_speedup_onnx, verbose=False, opset_version=11)
    print('pruned model exported.')
    flops, params, _ = count_flops_params(tmp_model, dummy_input, verbose=True)
    print(f"Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")


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

    device = "cuda:0"
    batch_size = 2
    width_mult = 1.0
    num_classes = 20
    input_size = [1, 3, 224, 224]
    model = resnet18(pretrained=True)
    # model = SimpleModel()
    # model = MobileNetV2()
    # model = build_model.get_models(net_type, input_size, num_classes, width_mult=width_mult, pretrained=False)
    model.eval()
    inputs = torch.randn(input_size)
    model = model.to((device))
    inputs = inputs.to((device))
    out = model(inputs)
    prune_model = copy.deepcopy(model)
    prune_model = model_pruning(prune_model, input_size=input_size, sparsity=0.9, dependency_aware=True, device=device)



