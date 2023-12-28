# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-26 15:37:56
    @Brief  : https://blog.csdn.net/JianguoChow/article/details/124637324
"""

import os
import torch.onnx
import onnx


def convert2onnx(model, input_shape, input_names=['input'], output_names=['output'],
                 onnx_file="", dynamic=False, simplify=True, opset_version=11,
                 use_prune=False, sparsity=0.2, device="cuda:0"):
    """
    :param model: Pytorch 模型
    :param input_shape: 输入维度(B, C, H, W)
    :param input_names: 输入节点名称
    :param output_names: 输出节点名称
    :param onnx_file: 输出ONNX模型文件
    :param dynamic:
    :param simplify: 是否对ONNX进行simplify
    :param opset_version: ONNX版本，9，11,建议使用11，版本9可能有异常
    :param use_prune: 是否对模型进行剪枝
    :param sparsity: 对模型进行剪枝的稀疏度
    :param device: 运行设备
    :return:
    """
    if onnx_file:
        output = os.path.dirname(onnx_file)
    else:
        output = "./output"
        onnx_file = os.path.join(output, "model.onnx")
    if not os.path.exists(output): os.makedirs(output)
    B, C, H, W = input_shape
    if use_prune:
        from basetrainer.pruning import nni_pruning
        model = nni_pruning.model_pruning(model,
                                          input_size=[8, 3, H, W],
                                          sparsity=sparsity,
                                          reuse=True,
                                          output_prune=os.path.join(output, "prune"),
                                          device="cuda")
    model = model.to(device)
    model.eval()
    if dynamic:
        inputs = torch.randn(B, C, H, W).to(device)
        # 声明动态维度，这里我们把input的第0维度赋名为batch_size
        dynamic_axes = {input_names[0]: {0: 'batch_size'}, output_names[0]: {0: 'batch_size'}}
    else:
        inputs = torch.randn(1, C, H, W).to(device)
        dynamic_axes = None
    do_constant_folding = True
    torch.onnx.export(model, inputs,
                      onnx_file,
                      verbose=False,
                      export_params=True,
                      do_constant_folding=do_constant_folding,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=opset_version)

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print("save onnx modle file:{}".format(onnx_file))
    # simplify model
    if simplify:
        try:
            import onnxsim
            print(f'simplifying with onnx-simplifier {onnxsim.__version__}')
            onnx_model, check = onnxsim.simplify(onnx_model, dynamic_input_shape=dynamic, input_shapes=None)
            onnx.save(onnx_model, onnx_file)
            print("simplifier onnx model:{}".format(onnx_file))
        except Exception as e:
            print(f'simplifier failure: {e}')
    return onnx_file


if __name__ == "__main__":
    pass
