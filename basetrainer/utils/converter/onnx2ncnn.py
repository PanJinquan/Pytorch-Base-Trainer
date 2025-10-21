# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-04-29 09:13:09
# @Brief  : https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx
#           https://github.com/pnnx/pnnx
# --------------------------------------------------------
"""

import os
import torch.onnx
import pnnx
import subprocess


def convert2ncnn(onnx_file, shape1, shape2=[], out_file="", fp16=True, use_prune=False, sparsity=0.2, device="cpu"):
    """
        output, name = os.path.split(opt.model_file)
    out_file = os.path.join(output, "ncnn", name.replace(".pth", ".jit"))

    https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx
    command: 默认开启fp16=1
    >>  pnnx model.onnx inputshape=[1,3,320,320]= pnnx model.onnx inputshape=[1,3,320,320]f32 fp16=1

    :param onnx_file: onnx 模型文件
    :param input_shape: 输入维度(B, C, H, W)
    :param input_names: 输入节点名称
    :param output_names: 输出节点名称
    :param out_file: 输出模型文件     out_file = model_file.replace(".pth", ".jit").replace(".pt", ".jit")
    :param use_prune: 是否对模型进行剪枝
    :param sparsity: 对模型进行剪枝的稀疏度
    :param device: 运行设备
    :return:
    """
    shape1 = list(shape1)
    shape2 = list(shape2) if shape2 else shape1
    cmd = ["pnnx",
           onnx_file,
           'fp16=0',
           f'inputshape="{shape1}"',
           f'inputshape2="{shape2}"',
           ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return True


if __name__ == "__main__":
    onnx_file = "data/model/resnet/resnet18_224_224.onnx"
    convert2ncnn(onnx_file, shape1=[1, 3, 224, 224])
