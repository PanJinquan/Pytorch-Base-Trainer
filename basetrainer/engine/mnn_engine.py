# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : PKing
# @E-mail : pan_jinquan@163.com
# @Date   : 2024-02-18 16:10:49
# @Brief  :  https://www.yuque.com/mnn/en/usage_in_python
             https://mnn-docs.readthedocs.io/en/latest/start/python.html
             模型转换工具 https://github.com/alibaba/MNN/wiki/convert#%25
             pip install mnn(不支持opencl等加速)
# --------------------------------------------------------
"""
import os, sys

sys.path.insert(0, os.getcwd())
import numpy as np
import MNN
from basetrainer.engine.onnx_engine import simplify_onnx, onnx_fp16, print_tensor
from basetrainer.utils.converter import onnx2mnn


class MNNEngine(object):
    def __init__(self, model_file, quant=0, simplify=False, dynamic=True, num_thread=4, device="cpu", **kwargs):
        """
        pip install --upgrade docs/MNN/vulkan/mnn-3.2.5-cp310-cp310-linux_x86_64.whl numpy==1.26.0 --force-reinstall
        pip install --upgrade docs/MNN/opencl/MNN-3.2.5-cp310-cp310-linux_x86_64.whl numpy==1.26.0 --force-reinstall
        CPU, OPENCL, OPENGL, NN, VULKAN, METAL, TRT, CUDA, HIAI
        config 中需要配置如下参数，均传整数，具体用法参考后面章节,详细见
        https://mnn-docs.readthedocs.io/en/latest/start/python.html
        https://github.com/alibaba/MNN/blob/master/docs/start/python.md
        backend    0:CPU     1:Metal 2:CUDA  3:OPENCL  5: NPU   7: VULKAN
        precision  0:normal(fp16存储，转换到fp32计算) 1:high(fp32存储和计算)  2:low(fp16存储和计算)
        memory     0:normal  1:high  2:low    0/1:权重量化的模型，加载时将权重反量化为浮点
        power      0:normal  1:high  2:low    目前仅高通的GPU支持调节
        ---------------------------------------------------------------------
        量化方法		              config            耗时 / ms	内存 / mb
        ---------------------------------------------------------------------
        FP32	            precision = 1, memory = 1	8.106100	19.242306
        基于FP32的动态量化	precision = 1, memory = 2	4.739200	9.624172
        FP16	            precision = 2, memory = 1	4.225200	9.762356
        基于FP16的动态量化	precision = 2, memory = 2	3.663600	6.616970
        ---------------------------------------------------------------------
        :param model_file:
        :param use_gpu: 是否使用GPU
        :param quant: 0:不进行量化，1:进行半精度量化(FP16)，2:进行INT8量化(INT8)
        :param simplify: 是否简化模型
        :param dynamic: 是否动态输入
        :param device: ["CPU", "CUDA", "OPENCL", "VULKAN"]
        """
        self.quant = quant
        self.simplify = simplify
        self.dynamic = dynamic
        self.device = device.upper()
        if model_file.endswith(".onnx"):  # TODO 如果是ONNX模型，需要转换为MNN模型
            if self.simplify: model_file = simplify_onnx(model_file, onnx_model=None, dynamic=dynamic)
            model_file = onnx2mnn.convert2mnn(model_file, fp16=self.quant == 1)
        assert os.path.exists(model_file), f"model file not exists:{model_file}"
        self.config = {
            "backend": self.device, # 后端idx,或者写设备名称CPU/CUDA/OPENCL/VULKAN
            "precision": 2 if self.quant == 1 else 0,
            "numThread": num_thread,
            "memory": 1,
            "power": 0,# 目前仅高通的GPU支持调节
        }
        # TODO
        rt = MNN.nn.create_runtime_manager((self.config,))
        # rt.set_cache(model_file.replace('.mnn', '.cache')) # 缓存模型文件，切换backend容易报错
        rt.set_mode(8) # Session_Backend_Fix = 8 Session_Backend_Auto = 9
        rt.set_hint(0, 10) # tune_num = 20
        # TODO MNN.Interpreter（传统推理接口），MNN.nn.load_module_from_file（高级模块接口）推荐使用后者
        self.inp_names, self.out_names = self.get_node_names(model_file)
        # 若输入shape固定，应设 shape_mutable=False 以提升性能。
        self.model = MNN.nn.load_module_from_file(file_name=model_file,
                                                  input_names=self.inp_names,
                                                  output_names=self.out_names,
                                                  shape_mutable=dynamic,
                                                  runtime_manager=rt,
                                                  )
        print("use device         :{}".format(self.device))
        print("inp_names          :{}".format(self.inp_names))
        print("out_names          :{}".format(self.out_names))
        print("quant level        :{}".format(self.quant))
        print("model_file         :{}".format(model_file))
        print('-----------' * 5, flush=True)

    @staticmethod
    def get_node_names(mnn_file):
        """
        获取MNN模型的输入输出节点名称
        :param mnn_file:
        :return:
        """
        interpreter = MNN.Interpreter(mnn_file)
        session = interpreter.createSession()
        inp_names = interpreter.getSessionInputAll(session)
        out_names = interpreter.getSessionOutputAll(session)
        # for name, tensor in inp_names.items():
        #     pass
        # for name, tensor in out_names.items():
        #     pass
        del interpreter
        return list(inp_names.keys()), list(out_names.keys())

    def __call__(self, image_tensor):
        """
        num_bboxes=1500
        num_class=1
        :param image_tensor: shape
        :return: outputs[0]
        """
        outputs = self.forward(image_tensor)
        return outputs

    def forward(self, inp_tensor: np.ndarray):
        """
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param inp_tensor:
        :return:
        """
        # inp_var = MNN.expr.const(inp_tensor, inp_tensor.shape, MNN.expr.NCHW)
        inp_vars = MNN.expr.const(inp_tensor, inp_tensor.shape, dtype=MNN.expr.float)
        out_vars = self.model.onForward([inp_vars])
        out_tensor = [var.read() for var in out_vars]  # List[np.ndarray
        return out_tensor

    def performance(self, inputs, iterate=50):
        from pybaseutils import time_utils
        outputs = self.forward(inputs)
        for i in range(iterate):
            with time_utils.Performance(n=2) as p:
                outputs = self.forward(inputs)
        return outputs


if __name__ == "__main__":
    import cv2
    from pybaseutils import image_utils

    model_file = "data/model/resnet/resnet18_224_224.mnn"
    # model_file = "data/model/yolov8n-seg.mnn"
    # model_file = "data/model/yolov8n-seg.onnx"

    input_shape = [1, 3, 224, 224]
    # np.random.seed(2020)
    # inputs = np.random.randn(*input_shape).astype(np.float32)
    filename = "data/test.jpg"
    image = cv2.imread(filename)
    image = cv2.resize(image, (224, 224))
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    inputs = (image.astype(np.float32) / 255.0 - mean) / std
    inputs = inputs.transpose(2, 0, 1)[np.newaxis, :]
    # CPU/CUDA/OPENCL/VULKAN
    model = MNNEngine(model_file, quant=1, simplify=False, device="VULKAN", dynamic=True, op_block=['Cast'])
    output = model.forward(inputs)
    model.performance(inputs)
    # print_tensor("inputs{}".format(inputs.shape), inputs[0, 0, 0, 0:20])
    # print_tensor("inputs{}".format(inputs.shape), inputs)
    print(output)
    # print_tensor("output", output, num=10)
    print(model_file)
