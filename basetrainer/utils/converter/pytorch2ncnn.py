# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-26 15:37:56
    @Brief  : https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx
              https://github.com/pnnx/pnnx
"""

import os
import torch.onnx
import pnnx


def convert2ncnn(model, input_shape, input_names=['input'], output_names=['output'],
                 out_file="", fp16=True, use_prune=False, sparsity=0.2, device="cpu"):
    """
    https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx
    command: 默认开启fp16=1
    >>  pnnx model.onnx inputshape=[1,3,320,320]= pnnx model.onnx inputshape=[1,3,320,320]f32 fp16=1

    :param model: Pytorch 模型
    :param input_shape: 输入维度(B, C, H, W)
    :param input_names: 输入节点名称
    :param output_names: 输出节点名称
    :param out_file: 输出模型文件     out_file = model_file.replace(".pth", ".jit").replace(".pt", ".jit")
    :param use_prune: 是否对模型进行剪枝
    :param sparsity: 对模型进行剪枝的稀疏度
    :param device: 运行设备
    :return:
    """
    if out_file:
        output = os.path.dirname(out_file)
    else:
        output = "./output"
        out_file = os.path.join(output, "model.jit")
    if not os.path.exists(output): os.makedirs(output)
    B, C, H, W = input_shape
    if use_prune:
        from basetrainer.pruning import nni_pruning
        model = nni_pruning.model_pruning(model,
                                          input_size=[8, 3, H, W],
                                          sparsity=sparsity,
                                          reuse=True,
                                          output_prune=os.path.join(output, "prune"),
                                          device=device)
    model = model.to(device)
    model.eval()
    inputs = torch.randn(B, C, H, W).to(device)
    model_ = pnnx.export(model, out_file, inputs=inputs, fp16=fp16)
    print("save ncnn modle file:{}".format(os.path.dirname(out_file)))
    return output


if __name__ == "__main__":
    pass
