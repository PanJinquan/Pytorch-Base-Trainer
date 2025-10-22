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

sys.path.append(os.getcwd())
import numpy as np
import MNN
from basetrainer.engine.onnx_engine import simplify_onnx, onnx_fp16
from basetrainer.utils.converter import onnx2mnn


class MNNEngine(object):
    def __init__(self, mnn_file, quant=0, simplify=False, dynamic=True, num_thread=4, device="cpu", **kwargs):
        """
        pip install --upgrade docs/MNN/vulkan/mnn-3.2.5-cp310-cp310-linux_x86_64.whl numpy==1.26.0 --force-reinstall
        pip install --upgrade docs/MNN/opencl/MNN-3.2.5-cp310-cp310-linux_x86_64.whl numpy==1.26.0 --force-reinstall
        CPU, OPENCL, OPENGL, NN, VULKAN, METAL, TRT, CUDA, HIAI
        config 中需要配置如下参数，均传整数，具体用法参考后面章节
        backend    0 : CPU       1 : Metal      2 : CUDA    3 : OpenCL     5: NPU   7: Vulkan
        precision  0 : normal    1 : high       2 : low
        memory     0 : normal    1 : high       2 : low
        power      0 : normal    1 : high       2 : low
        :param mnn_file:
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
        if mnn_file.endswith(".onnx"):  # TODO 如何是ONNX模型，需要转换为MNN模型
            if self.simplify: mnn_file = simplify_onnx(mnn_file, onnx_model=None, dynamic=dynamic)
            mnn_file = onnx2mnn.convert2mnn(mnn_file, fp16=self.quant == 1)
        assert os.path.exists(mnn_file), f"*.mnn file not exists:{mnn_file}"
        self.config = {
            "backend": self.device,
            "precision": "low" if self.quant == 1 else "normal",
            "numThread": num_thread,
            "memory": "normal",
            "power": "normal",
        }
        # TODO
        rt = MNN.nn.create_runtime_manager((self.config,))
        rt.set_cache(mnn_file.replace('.mnn', '.cache'))
        # TODO MNN.Interpreter（传统推理接口），MNN.nn.load_module_from_file（高级模块接口）推荐使用后者
        self.inp_names, self.out_names = self.get_node_names(mnn_file)
        # 若输入shape固定，应设 shape_mutable=False 以提升性能。
        self.model = MNN.nn.load_module_from_file(file_name=mnn_file,
                                                  input_names=self.inp_names,
                                                  output_names=self.out_names,
                                                  shape_mutable=dynamic,
                                                  runtime_manager=rt,
                                                  )
        print("use device         :{}".format(self.device))
        print("inp_names          :{}".format(self.inp_names))
        print("out_names          :{}".format(self.out_names))
        print("quant level        :{}".format(self.quant))
        print("onnx_file          :{}".format(mnn_file))
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
    # mnn_file = "../../data/model/resnet/resnet18_224_224.mnn"
    # mnn_file = "../../data/model/yolov8n-seg.mnn"
    mnn_file = "../../data/model/yolov8n-seg.onnx"
    input_shape = [5, 3, 640, 640]
    np.random.seed(2020)
    inputs = np.random.randn(*input_shape).astype(np.float32)
    model = MNNEngine(mnn_file, quant=0, simplify=False, dynamic=True, op_block=['Cast'])
    output = model.forward(inputs)
    model.performance(inputs)
    print("inputs=", inputs[0, 0, 0, 0:20])
    print("output=", output)
    print("----")
