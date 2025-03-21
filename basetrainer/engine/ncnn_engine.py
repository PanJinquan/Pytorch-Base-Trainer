# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : PKing
# @E-mail : pan_jinquan@163.com
# @Date   : 2024-02-18 16:10:49
# @Brief  :
# --------------------------------------------------------
"""

import os
import sys
import ncnn
import numpy as np


class NCNNEngine(object):
    def __init__(self, par_file, bin_file="", num_thread=8, use_gpu=True, use_fp16=True):
        """
        pnnx教程：https://github.com/pnnx/pnnx
        ncnn教程：https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx
        YOLOv8:  https://github.com/jahongir7174/YOLOv8-onnx/tree/master
        NCNN推理引擎，需要注意：
        (1)将输入数组转换为C/C++数组时，需要保证输入数据是顺序（行优先）的连续数组，否则会出现推理异常等问题
           为避免内存不连续，建议使用np.ascontiguousarray(data)或者使用copy()
        (2)NCNN仅仅支持(c,h,w),不能输入(1,c,h,w),否则推理结果异常，建议使用np.squeeze(data)简化数组维度
        :param par_file: NCNN param file path (.param)
        :param bin_file: NCNN model weights file path (.bin)
        :param num_thread: CPU线程个数，-1表示等于系统cpu个数
        :param use_gpu: Whether to use GPU (Vulkan) for inference
        :param use_fp16: Whether to enable FP16 inference
        pip install ncnn
        """
        if not bin_file: bin_file = par_file.replace(".param", ".bin")
        assert os.path.exists(par_file), f"*.param file not exists:{par_file}"
        assert os.path.exists(bin_file), f"*.bin   file not exists:{bin_file}"
        self.cpu_count = ncnn.get_cpu_count()
        self.gpu_count = ncnn.get_gpu_count()
        self.gpu_info = ncnn.get_gpu_info()
        self.gpu_index = ncnn.get_default_gpu_index()
        self.device = "cpu"
        self.num_thread = num_thread if num_thread > 0 else self.cpu_count
        # TODO 创建ncnn的Net对象
        # ncnn.set_omp_num_threads(self.num_thread)
        self.net = ncnn.Net()
        self.net.opt.num_threads = self.num_thread  # must set net.opt.num_threads=N before net.load_param()
        if use_gpu:  # Enable vulkan compute if GPU is requested
            try:
                # ncnn.create_gpu_instance()
                self.net.opt.use_vulkan_compute = True
                self.net.set_vulkan_device(self.gpu_index)
                self.device = self.net.vulkan_device().info().device_name()
                # self.device = ncnn.get_gpu_device(self.gpu_index).info().device_name()  # 如果没有GPU会出现异常
            except Exception as e:
                print("Warning：当前环境不支持GPU运行，自动切换到CPU运行")
                self.net.opt.use_vulkan_compute = False
        if use_fp16:
            self.net.opt.use_fp16_packed = True
            self.net.opt.use_fp16_storage = True
            self.net.opt.use_fp16_arithmetic = True
        # Load model
        self.net.load_param(par_file)
        self.net.load_model(bin_file)
        # Get input and output names from net
        self.inp_names = list(sorted(self.net.input_names()))
        self.out_names = list(sorted(self.net.output_names()))
        print("use_device        :{}".format(self.device))
        print("cpu_info          :cpu_count={},use threads:{}".format(self.cpu_count, self.num_thread))
        print("gpu_info          :gpu_count={},gpu_info={}".format(self.gpu_count, self.gpu_info))
        print("use_gpu           :{},gpu_index={}".format(use_gpu, self.gpu_index))
        print("use_fp16          :{}".format(use_fp16))
        print("inp_names         :{}".format(self.inp_names))
        print("out_names         :{}".format(self.out_names))
        print("param_file        :{}".format(par_file))
        print("bin_file          :{}".format(bin_file))
        print('-----------' * 5, flush=True)

    def __call__(self, image_tensor):
        """
        :param image_tensor: numpy array with shape (b, c, h, w)
        :return: model outputs
        """
        outputs = self.forward(image_tensor)
        return outputs

    def forward(self, image_tensor):
        """
        :param image_tensor: numpy array with shape (b, c, h, w)
        :return: list of outputs
        """
        ex = self.net.create_extractor()
        # ex.set_num_threads(self.num_thread);
        # BUG：为避免内存不连续，需要copy或者np.ascontiguousarray()
        data = np.ascontiguousarray(np.squeeze(image_tensor))  # 使用np.squeeze(data)简化数组维度，避免结果异常
        # data = np.ascontiguousarray(image_tensor)    # BUG: 这样写，推理结果有异常，没有squeeze
        # input = ncnn.Mat(np.ascontiguousarray(data)) # BUG: 这样写，推理结果有异常
        input = ncnn.Mat(data)  # 推理结果正常
        ex.input(self.inp_names[0], input)
        # Extract outputs
        outputs = []
        for name in self.out_names:
            _, out = ex.extract(name)
            outputs.append(np.array(out).reshape(1, out.c, out.h, out.w))
            out.release()
        input.release()
        return outputs

    def forward_image(self, image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        """
        :param image: numpy array with shape (h, w,3)
        :return: list of outputs
        """
        mean = np.array(mean, dtype=np.float32) * 255
        std = 1 / np.array(std, dtype=np.float32) / 255
        with self.net.create_extractor() as ex:
            # Convert numpy array to ncnn.Mat
            data = np.ascontiguousarray(np.squeeze(image))  # BUG：内存不连续，需要copy或者np.ascontiguousarray()
            input = ncnn.Mat.from_pixels(data, ncnn.Mat.PixelType.PIXEL_RGB, data.shape[1], data.shape[0])  # (w,h)
            input.substract_mean_normalize(mean, std);
            ex.input(self.inp_names[0], input.clone())
            # Extract outputs
            outputs = []
            for name in self.out_names:
                _, out = ex.extract(name)
                outputs.append(np.array(out).reshape(1, out.c, out.h, out.w))
        return outputs

    def performance(self, inputs, iterate=20):
        """
        :param inputs: inputs (1,c,h,w)
        :param iterate: number of iterations for performance testing
        :return: outputs
        """
        from pybaseutils import time_utils
        outputs = self.forward(inputs)
        for i in range(iterate):
            with time_utils.Performance() as p:
                outputs = self.forward(inputs)
        print("inp=\n", inputs[0, 0, 0, 0:10])
        print("out=")
        [print(out[0, 0, 0, 0:10]) for out in outputs]
        return outputs


if __name__ == "__main__":
    np.random.seed(100)
    # Example usage
    # param_file = "../../data/model/ncnn-fp16/model.ncnn.param"
    param_file = "../../data/model/ncnn-fp32/model.ncnn.param"
    # Create random input
    input_size = [168, 168]
    inputs = np.random.random(size=(1, 3, input_size[1], input_size[0]))
    inputs = np.asarray(inputs, dtype=np.float32)
    model = NCNNEngine(param_file, use_gpu=True, use_fp16=True)
    model.performance(inputs)
    print("----")
