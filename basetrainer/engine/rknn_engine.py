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

sys.path.insert(0, os.getcwd())
import numpy as np
import platform
from basetrainer.engine.onnx_engine import simplify_onnx, onnx_fp16, print_tensor


class RKNNEngine(object):
    def __init__(self,
                 model_file,
                 shape=(1, 3, 224, 224),
                 device=None,
                 quant=0,
                 simplify=True,
                 dynamic=True,
                 dynamic_shape=[],
                 **kwargs):
        """
        安装RKNN-Toolkit2（PC端）或RKNN-Toolkit-Lite2（板端）
        - PC端：pip install docs/rknn-toolkit2/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
        - 板端：pip install docs/rknn-toolkit2/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
        :param model_file:
        :param device: rknn目标平台，simulator or rv1103/rv1103b/rv1106/rv1106b/rv1126b/rk3562/rk3566/rk3568/rk3576/rk3588.
                       default is None, means simulator.
        :param quant: 0:不进行量化(FP16)，1: 混合精度量化，2:进行INT8量化(INT8)
                      RKNPU目前不支持FP32的计算方式，因此模拟器在不开启量化的情况下，默认是FP16的运算类型，所以
                      只需要在使用rknn.build()接口时，将do_quantization参数设置为False，即可以将原始模型转换为FP16的
                      RKNN模型，接着调用rknn.init_runtime(target=None)和rknn.inference()接口进行FP16模拟推理并获取输出结果。
        :param simplify: 是否简化模型，建议True
        :param dynamic: 是否动态输入, True: CPU模式逐个推理，比批量推理快
        :param dynamic_shape: [[[1, 3, 640, 640]],[[1, 3, 480, 480]], [[1, 3, 320, 320]]]
        :param kwargs: 其他参数，如op_block=['Cast'], nd_block等
        TODO
        """
        if not dynamic_shape: dynamic_shape = [[shape]]
        self.host_name = get_system_host_name()  # rk3588 rk3566 rk3568
        self.quant = quant
        self.simplify = simplify
        self.dynamic = dynamic
        self.device = device
        assert os.path.exists(model_file), f"model file not exists:{model_file}"
        self.model = None
        if model_file.endswith(".onnx"):  # TODO 如果是ONNX模型，需要转换为MNN模型
            print("load onnx model        :{}".format(model_file))
            if self.simplify: model_file = simplify_onnx(model_file, onnx_model=None, dynamic=dynamic)
            # if self.quant == 1:
            #     model_file = onnx_fp16(model_file, out_file="", **kwargs)
            # TODO 转换为RKNN模型
            model_file, self.model = self.export_rknn(model_file, shapes=dynamic_shape, quant=quant, **kwargs)
        # TODO 如果是RK机器，则加载RKNN模型
        if self.host_name.startswith("rk") and model_file.endswith(".rknn"):
            assert os.path.exists(model_file), f"model file not exists:{model_file}"
            print("load rknn model        :{}".format(model_file))
            self.model = self.build_rknn_lite(model_file, host_name=self.host_name)
        self.inp_names = ""
        self.out_names = ""
        print("use device         :{}".format(self.device))
        print("host  name         :{}".format(self.host_name))
        print("inp_names          :{}".format(self.inp_names))
        print("out_names          :{}".format(self.out_names))
        print("quant level        :{}".format(self.quant))
        print("model_file         :{}".format(model_file))
        print('-----------' * 5, flush=True)

    @staticmethod
    def export_rknn(onnx_file, rknn_file="", shapes=[1, 3, 224, 224], device=None, quant=0,
                    dataset="./images.txt", **kwargs):
        """
        TODO pip install rknn-toolkit2
        :param onnx_file:
        :param rknn_file:
        :param shapes: [[[1, 3, 640, 640]],[[1, 3, 480, 480]], [[1, 3, 320, 320]]]
        :param quant: 0:不进行量化(FP16)，1: 混合精度量化，2:进行INT8量化(INT8)
        :param dataset: dataset: find images/ -type f > images.txt, 包含所有图像的txt文件
        :return: rknn.release(), rknn_file
        """
        # TODO 详细看02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_CN.pdf
        from rknn.api import RKNN
        rknn_file = rknn_file if rknn_file else onnx_file.replace(".onnx", ".rknn")
        rknn_file = rknn_file.replace(".rknn", "_int8.rknn") if quant else rknn_file
        rknn = RKNN(verbose=True)
        device = device if device else "rk3588"
        # 预处理器置
        rknn.config(mean_values=kwargs.get("mean_values", None),
                    std_values=kwargs.get("std_values", None),
                    target_platform=device,  # 请根据你的开发板芯片修改，如rk3568, rk3588
                    dynamic_input=shapes if shapes else None
                    )
        # 加载ONNX模型
        ret = rknn.load_onnx(model=onnx_file, input_size_list=shapes if shapes else None)
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        # 构建RKNN模型,INT8量化可以显著提升在板端的推理性能，建议开启
        ret = rknn.build(do_quantization=quant > 0,  # 是否进行量化
                         auto_hybrid=quant == 1,  # 自动混合精度量化
                         dataset=dataset, )
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        # 导出RKNN模型文件
        ret = rknn.export_rknn(rknn_file)
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        # 释放资源
        # rknn.release()
        print("Export rknn model success,rknn={}，shapes=:{}".format(rknn_file, shapes))
        ret = rknn.init_runtime(target=None)
        return rknn_file, rknn

    @staticmethod
    def build_rknn_lite(rknn_file="", host_name=None):
        """
        TODO
        :param rknn_file:
        :param quant:
        :param dataset: dataset: find images/ -type f > images.txt, 包含所有图像的txt文件
        :return: rknn.release(), rknn_file
        """
        from rknnlite.api import RKNNLite
        rknn_lite = RKNNLite()
        # Load RKNN model
        print('--> Load RKNN model,host_name={}'.format(host_name))
        ret = rknn_lite.load_rknn(rknn_file)
        if ret != 0:
            print('Load RKNN model failed')
            exit(ret)
        print('--> Init runtime environment')
        # Run on RK356x / RK3576 / RK3588 with Debian OS, do not need specify target.
        if host_name in ['rk3576', 'rk3588']:
            # For RK3576 / RK3588, specify which NPU core the model runs on through the core_mask parameter.
            ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        else:
            ret = rknn_lite.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        return rknn_lite

    def get_out_names(self):
        names = []
        return names

    def get_inp_names(self, ):
        names = []
        return names

    def __call__(self, image_tensor):
        """
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
        # nchw转换为nhwc，RKNN模型输入要求为nhwc
        inp_tensor = np.transpose(inp_tensor, (0, 2, 3, 1))
        # data_format: Data format list, current support: 'nhwc', 'nchw', default is 'nhwc', only valid for 4-dims input. default is None.
        out_tensor = self.model.inference(inputs=[inp_tensor])
        return out_tensor

    def performance(self, inputs, iterate=50):
        from pybaseutils import time_utils
        outputs = self.forward(inputs)
        for i in range(iterate):
            with time_utils.Performance(n=2) as p:
                outputs = self.forward(inputs)
        return outputs


def get_system_host_name():
    """
    获取系统机器类型
    :return:
    """
    # decice tree for RK356x/RK3576/RK3588
    DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3562' in device_compatible_str:
                    host = 'rk3562'
                elif 'rk3576' in device_compatible_str:
                    host = 'rk3576'
                elif 'rk3588' in device_compatible_str:
                    host = 'rk3588'
                else:
                    host = 'rk3566_rk3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host.lower()


if __name__ == "__main__":
    # model_file = "../../data/model/resnet18_224_224.onnx"
    # model_file = "data/model/yolov8n-seg.onnx"
    # model_file = "data/model/yolov8n-seg.rknn"
    model_file = "data/model/yolov8n-seg_int8.rknn"
    input_shape = [1, 3, 640, 640]
    dynamic_shape = [[[1, 3, 640, 640]], [[1, 3, 480, 480]], [[1, 3, 320, 320]]]
    np.random.seed(2020)
    inputs = np.random.randint(0, 255, size=input_shape).astype(np.float32)
    inputs = inputs.astype(np.uint8)
    model = RKNNEngine(model_file, shape=input_shape, quant=0, simplify=False, device="rk3588",
                       dynamic_shape=dynamic_shape)
    output = model.forward(inputs)
    model.performance(inputs)
    print_tensor("inputs{}".format(inputs.shape), inputs[0, 0, 0, 0:20])
    print_tensor("output", output, num=10)
    print(model_file)
