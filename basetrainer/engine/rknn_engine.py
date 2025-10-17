# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : PKing
# @E-mail : pan_jinquan@163.com
# @Date   : 2024-02-18 16:10:49
# @Brief  : https://docs.radxa.com/rock5/rock5itx/app-development/rknn_install
            https://github.com/airockchip/rknn-toolkit2
# --------------------------------------------------------
"""
import os, sys

sys.path.append(os.getcwd())
import numpy as np
from collections import defaultdict
from basetrainer.utils.converter.pytorch2onnx import onnx_fp16, simplify_onnx


class RKNNEngine(object):
    def __init__(self,
                 onnx_file,
                 shape=(1, 3, 224, 224),
                 use_npu=True,
                 device=None,
                 quant=0,
                 simplify=False,
                 dynamic=True,
                 **kwargs):
        """
        安装RKNN-Toolkit2（PC端）或RKNN-Toolkit-Lite2（板端）
        PC端：pip install rknn-toolkit2
        板端：sudo apt install rknpu2-rk3588
             sudo apt install python3-rknnlite2
        :param onnx_file:
        :param use_npu: 是否使用GPU
        :param device: rknn目标平台，simulator or rv1103/rv1103b/rv1106/rv1106b/rv1126b/rk3562/rk3566/rk3568/rk3576/rk3588.
                       default is None, means simulator.
        :param quant: 0:不进行量化，1:进行半精度量化(FP16)，2:进行INT8量化(INT8)
        :param simplify: 是否简化模型
        :param dynamic: 是否动态输入, True: CPU模式逐个推理，比批量推理快
        :param kwargs: 其他参数，如op_block=['Cast'], nd_block等
        TODO
        """
        self.quant = quant
        self.simplify = simplify
        self.dynamic = dynamic
        self.device = device
        if self.simplify:
            onnx_file = simplify_onnx(onnx_file, onnx_model=None, dynamic=dynamic)
        # if self.quant == 1:
        #     onnx_file = onnx_fp16(onnx_file, out_file="", **kwargs)
        assert os.path.exists(onnx_file), f"*.onnx file not exists:{onnx_file}"
        self.rknn_file, self.model = self.export_rknn(onnx_file, shape=shape, device=self.device,
                                                      quant=bool(quant), **kwargs)
        # self.model = self.build_rknn_lite(self.rknn_file)
        self.inp_names = ""
        self.out_names = ""
        print("use device         :{}".format(self.device))
        print("inp_names          :{}".format(self.inp_names))
        print("out_names          :{}".format(self.out_names))
        print("quant level        :{}".format(self.quant))
        print("onnx_file          :{}".format(onnx_file))
        print('-----------' * 5, flush=True)

    @staticmethod
    def export_rknn(onnx_file, rknn_file="", shape=[1, 3, 224, 224], device=None,
                    quant=False, dataset="./images.txt"):
        """
        TODO pip install rknn-toolkit2
        :param onnx_file:
        :param rknn_file:
        :param shape:
        :param quant:
        :param dataset: dataset: find images/ -type f > images.txt, 包含所有图像的txt文件
        :return: rknn.release(), rknn_file
        """
        from rknn.api import RKNN
        shape = list(shape)
        rknn_file = rknn_file if rknn_file else onnx_file.replace(".onnx", ".rknn")
        rknn = RKNN(verbose=True)
        device = device if device else "rk3566"
        # 预处理器置
        rknn.config(target_platform=device,  # 请根据你的开发板芯片修改，如rk3568, rk3588
                    dynamic_input=[[shape]] if shape else None
                    )
        # 加载ONNX模型
        ret = rknn.load_onnx(model=onnx_file,
                             input_size_list=[shape] if shape else None
                             )
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        # 构建RKNN模型,INT8量化可以显著提升在板端的推理性能，建议开启:cite[3]
        ret = rknn.build(do_quantization=quant, dataset=dataset)
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        # # 导出RKNN模型文件
        ret = rknn.export_rknn(rknn_file)
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        # 释放资源
        # rknn.release()
        print("Export rknn model success! rknn_file:{}".format(rknn_file))
        ret = rknn.init_runtime(target=None)
        return rknn_file, rknn

    @staticmethod
    def build_rknn_lite(rknn_file="", shape=[1, 3, 224, 224], device=None):
        """
        TODO
        sudo apt install rknpu2-rk3588
        sudo apt install python3-rknnlite2
        https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages
        :param rknn_file:
        :param shape:
        :param quant:
        :param dataset: dataset: find images/ -type f > images.txt, 包含所有图像的txt文件
        :return: rknn.release(), rknn_file
        """
        from rknnlite.api import RKNNLite as RKNN
        shape = list(shape)
        device = device if device else "rk3566"
        rknn = RKNN(verbose=True)
        ret = rknn.load_rknn(model=rknn_file,
                             input_size_list=[shape] if shape else None
                             )
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        ret = rknn.init_runtime(target=device)
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        return rknn

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
        out_tensor = self.model.inference(inputs=[inp_tensor], data_format="nchw")
        return out_tensor

    def performance(self, inputs, iterate=50):
        from pybaseutils import time_utils
        outputs = self.forward(inputs)
        for i in range(iterate):
            with time_utils.Performance(n=2) as p:
                outputs = self.forward(inputs)
        return outputs


if __name__ == "__main__":
    from basetrainer.utils.converter.pytorch2onnx import onnx_fp16

    # onnx_file = "../../data/model/resnet18_224_224.onnx"
    onnx_file = "../../data/model/yolov8n-seg.onnx"
    input_shape = [1, 3, 640, 640]
    np.random.seed(2020)
    inputs = np.random.randn(*input_shape).astype(np.float32)
    model = RKNNEngine(onnx_file, use_npu=False, quant=0, simplify=False, dynamic=False,shape=input_shape)
    output = model.forward(inputs)
    model.performance(inputs)
    print("inputs=", inputs[0, 0, 0, 0:20])
    print("output=", output)
    print("----")
