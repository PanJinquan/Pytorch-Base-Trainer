# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : PKing
# @E-mail : pan_jinquan@163.com
# @Date   : 2024-02-18 16:10:49
# @Brief  : pip uninstall onnxruntime 先卸载cpu，然后再安装gpu版本
            pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
# --------------------------------------------------------
"""
import os, sys

sys.path.append(os.getcwd())
import onnx
import onnxruntime
import numpy as np
from collections import defaultdict


class ONNXEngine(object):
    def __init__(self, onnx_file, use_gpu=True, quant=0, simplify=False, dynamic=True, device_id=0, **kwargs):
        """
        pnnx教程：https://github.com/pnnx/pnnx
        ncnn教程：https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx
        YOLOv8:  https://github.com/jahongir7174/YOLOv8-onnx/tree/master
        :param onnx_file:
        :param use_gpu: 是否使用GPU
        :param quant: 0:不进行量化，1:进行半精度量化(FP16)，2:进行INT8量化(INT8)
        :param simplify: 是否简化模型
        :param dynamic: 是否动态输入, True: CPU模式逐个推理，比批量推理快
        :param device_id: GPU id
        :param kwargs: 其他参数，如op_block=['Cast'], nd_block等
        """
        self.quant = quant
        self.simplify = simplify
        self.dynamic = dynamic
        if self.simplify:
            onnx_file = simplify_onnx(onnx_file, onnx_model=None, dynamic=dynamic)
        if self.quant == 1:
            onnx_file = onnx_fp16(onnx_file, out_file="", **kwargs)
        assert os.path.exists(onnx_file), f"*.onnx file not exists:{onnx_file}"
        available_providers = onnxruntime.get_available_providers()
        options = onnxruntime.SessionOptions()
        if use_gpu:
            self.providers = [('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']
        # options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.onnx_session = onnxruntime.InferenceSession(onnx_file, providers=self.providers, sess_options=options)
        # self.device = onnxruntime.get_device()
        self.device = self.onnx_session.get_providers()
        self.inp_names = self.get_inp_names(self.onnx_session)
        self.out_names = self.get_out_names(self.onnx_session)
        print("available_providers:{}".format(available_providers))
        print("use device         :{}".format(self.device))
        print("inp_names          :{}".format(self.inp_names))
        print("out_names          :{}".format(self.out_names))
        print("quant level        :{}".format(self.quant))
        print("onnx_file          :{}".format(onnx_file))
        print('-----------' * 5, flush=True)

    def get_out_names(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        names = []
        for node in onnx_session.get_outputs():
            names.append(node.name)
        # names = list(sorted(names))
        return names

    def get_inp_names(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        names = []
        for node in onnx_session.get_inputs():
            names.append(node.name)
            if node.type == 'tensor(float)':
                self.quant = 0
            elif node.type == 'tensor(float16)':
                self.quant = 1
            elif node.type == 'tensor(int)':
                self.quant = 2
        # names = list(sorted(names))
        return names

    def get_inp_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def __call__(self, image_tensor):
        """
        num_bboxes=1500
        num_class=1
        :param image_tensor: shape
        :return: outputs[0]
        """
        outputs = self.forward(image_tensor)
        return outputs

    def forward(self, inp_tensor):
        """
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param inp_tensor:
        :return:
        """
        if self.quant == 1:
            inp_tensor = inp_tensor.astype(np.float16)
        if self.dynamic and len(inp_tensor) > 1:
            out_tensor = defaultdict(list)  # TODO  CPU模式逐个推理，比批量推理快
            for i in range(len(inp_tensor)):
                inp_feed = self.get_inp_feed(self.inp_names, inp_tensor[i:i + 1, ...])
                out = self.onnx_session.run(self.out_names, input_feed=inp_feed)
                for k in range(len(out)):
                    out_tensor[k].append(out[k])
            out_tensor = [np.concatenate(v, axis=0) for k, v in out_tensor.items()]
        else:
            # 输入数据的类型必须与模型一致,以下三种写法都是可以的
            # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
            # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
            inp_feed = self.get_inp_feed(self.inp_names, inp_tensor)
            out_tensor = self.onnx_session.run(self.out_names, input_feed=inp_feed)
        return out_tensor

    def performance(self, inputs, iterate=50):
        from pybaseutils import time_utils
        outputs = self.forward(inputs)
        for i in range(iterate):
            with time_utils.Performance(n=2) as p:
                outputs = self.forward(inputs)
        return outputs


def simplify_onnx(onnx_file: str, onnx_model=None, dynamic=False):
    """
    简化ONNX模型
    :param onnx_file:
    :param onnx_model:
    :return:
    """
    onnx_model = onnx_model if onnx_model else onnx.load(onnx_file)
    try:
        import onnxsim
        print(f'simplifying with onnx-simplifier {onnxsim.__version__}')
        onnx_model, check = onnxsim.simplify(onnx_model, dynamic_input_shape=dynamic, input_shapes=None)
        # import onnxslim
        # print(f'simplifying with onnx-simplifier {onnxslim.__version__}')
        # onnx_model = onnxslim.slim(onnx_model)
        out_file = onnx_file.replace(".onnx", "_sim.onnx")
        onnx.save(onnx_model, out_file)
        print("simplifier onnx model:{}".format(out_file))
    except Exception as e:
        print(f'simplifier failure: {e}')
        out_file = None
    return out_file


def onnx_fp16(onnx_file: str, onnx_model=None, out_file="", op_block=[], nd_block=[]):
    """
    pip install onnxconverter-common
    将FP32模型转换为FP16模型
    :param onnx_file:
    :param onnx_model:
    :param op_block: 指定哪些算子不转换为FP16，如["Softmax", "BatchNormalization"]
    :param nd_block: 指定哪些节点类型不转换为FP16，如["layer1/attention/weight", "output_layer/scale"]
    :return:
    """
    from onnxconverter_common import float16
    try:
        fp32_model = onnx_model if onnx_model else onnx.load(onnx_file)
        # fp16_model = float16.convert_float_to_float16(fp32_model, op_block_list=['Cast', 'Resize']) # 会卡死
        fp16_model = float16.convert_float_to_float16(fp32_model)
        out_file = out_file if out_file else onnx_file.replace(".onnx", "_fp16.onnx")
        onnx.save(fp16_model, out_file)
        out_file = fix_onnx_fp16(out_file, onnx_model, out_file=out_file, op_block=op_block, nd_block=nd_block)
        print(f'converter model FP16 success: {out_file}')
    except Exception as e:
        fp16_model = None
        out_file = None
        print(f'converter model FP16 failure: {e}')
    return out_file


def fix_onnx_fp16(onnx_file: str, onnx_model=None, out_file="", op_block=[], nd_block=[]):
    """
    修复ONNX模型
    :param onnx_file:
    :param onnx_model:
    :return:
    """
    if len(op_block) == 0 and len(nd_block) == 0: return onnx_file
    out_file = out_file if out_file else onnx_file.replace(".onnx", "_fix.onnx")
    model = onnx_model if onnx_model else onnx.load(onnx_file)

    def match_op_name(op_list, tar):
        """
        使用循环判断列表中是否存在目标字符串
        """
        for op in op_list:
            if op in tar:  return op
        return ""

    print('-----------' * 5, flush=True)
    print(f'fixed model FP16: {out_file},op_block={op_block}, nd_block={nd_block}')
    # 找到有问题的节点并修改其输出类型
    for node in model.graph.node:
        op_name = match_op_name(op_block, node.name)
        if op_name or (node.name in nd_block):
            print("op:{:10s},node:{:50s},float16 --> float32".format(node.op_type, node.name))
            for output in node.output:  # 找到对应的输出
                for value_info in model.graph.value_info:
                    if value_info.name == output:
                        # 修改类型为 float
                        value_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, out_file)
    print(f'fixed model FP16 success    : {out_file}')
    print('-----------' * 5, flush=True)
    return out_file


if __name__ == "__main__":
    from basetrainer.utils.converter.pytorch2onnx import onnx_fp16

    # onnx_file = "../../data/model/resnet18_224_224.onnx"
    onnx_file = "../../data/model/yolov8n-seg.onnx"
    input_shape = [1, 3, 640, 640]
    np.random.seed(2020)
    inputs = np.random.randn(*input_shape).astype(np.float32)
    model = ONNXEngine(onnx_file, use_gpu=False, quant=1, simplify=False, dynamic=False, op_block=['Cast'])
    output = model.forward(inputs)
    model.performance(inputs)
    print("inputs=", inputs[0, 0, 0, 0:20])
    # print("output=", output)
    print("----")
