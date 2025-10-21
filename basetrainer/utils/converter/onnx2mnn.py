# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-04-29 09:13:09
# @Brief  : 模型转换工具 https://github.com/alibaba/MNN/wiki/convert#%25
# --------------------------------------------------------
"""
import os
import subprocess
import MNN


def convert2mnn(onnx_file, shape1=[], shape2=[], out_file="", fp16=False):
    """
    模型转换工具 https://github.com/alibaba/MNN/wiki/convert#%25
    :param onnx_file: onnx 模型文件
    :param shape1: 输入维度(B, C, H, W)
    :param shape2: 输入节点名称
    :param out_file: 输出模型文件     out_file = onnx_file.replace(".onnx", ".mnn")
    :return:
    """
    out_file = out_file if out_file else onnx_file.replace(".onnx", ".mnn")
    cmd = ["mnnconvert",
           "-f", "ONNX",
           "--modelFile", onnx_file,
           "--MNNModel", out_file,
           "--fp16" if fp16 else None,
           "--info"
           ]
    cmd = [c for c in cmd if c]
    print("cmd: ", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, check=True)
    if r.returncode == 0:
        print(f"✅ Conversion successful: {out_file}")
        print("save mnn model to {}".format(os.path.abspath(out_file)))
    else:
        print("❌ Conversion failed:")
        print(r.stderr)
        raise RuntimeError("MNN conversion failed")
    return out_file


if __name__ == "__main__":
    # onnx_file = "data/model/resnet/resnet18_224_224.onnx"
    onnx_file = "../../../data/model/yolov8n-seg.onnx"
    convert2mnn(onnx_file, shape1=[1, 3, 640, 640])
