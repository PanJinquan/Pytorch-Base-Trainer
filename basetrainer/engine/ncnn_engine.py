# -*- coding: utf-8 -*-
"""
    @Author : Pan
    @Brief  : NCNN inference engine
"""
import os
import sys
import ncnn
import numpy as np
from pybaseutils import time_utils


class NCNNEngine(object):
    def __init__(self, param_file, bin_file="", inp_names=['in0'], out_names=['out0'], use_gpu=False, use_fp16=False):
        """
        :param param_file: NCNN param file path (.param)
        :param bin_file: NCNN model weights file path (.bin)
        :param use_gpu: Whether to use GPU (Vulkan) for inference
        :param use_fp16: Whether to enable FP16 inference
        pip install ncnn
        """
        if not bin_file: bin_file = param_file.replace(".param", ".bin")
        self.net = ncnn.Net()
        # Enable vulkan compute if GPU is requested
        if use_gpu:
            self.net.opt.use_vulkan_compute = True
            if use_fp16:
                self.net.opt.use_fp16_packed = True
                self.net.opt.use_fp16_storage = True
                self.net.opt.use_fp16_arithmetic = True

        # Load model
        self.net.load_param(param_file)
        self.net.load_model(bin_file)

        # Get input and output names from first inference
        self.inp_names = inp_names
        self.out_names = out_names
        print("use_gpu           :{}".format(use_gpu))
        print("use_fp16          :{}".format(use_fp16))
        print("inp_names         :{}".format(self.inp_names))
        print("out_names         :{}".format(self.out_names))
        print("param_file        :{}".format(param_file))
        print("bin_file          :{}".format(bin_file))
        print('-----------' * 5, flush=True)

    def __call__(self, image_tensor):
        """
        :param image_tensor: numpy array with shape (batch, channels, height, width)
        :return: model outputs
        """
        outputs = self.forward(image_tensor)
        return outputs

    def forward(self, image_tensor):
        """
        :param image_tensor: numpy array with shape (batch, channels, height, width)
        :return: list of outputs
        """
        with self.net.create_extractor() as ex:
            # Convert numpy array to ncnn.Mat
            mat_in = ncnn.Mat(image_tensor)
            ex.input(self.inp_names[0], mat_in)
            # Extract outputs
            outputs = []
            for name in self.out_names:
                _, output = ex.extract(name)
                outputs.append(output.numpy().reshape(1, output.c, output.h, output.w))
        return outputs

    def performance(self, inputs, iterate=10):
        """
        :param inputs: input tensor
        :param iterate: number of iterations for performance testing
        :return: outputs
        """
        outputs = self.forward(inputs)
        for i in range(iterate):
            with time_utils.Performance() as p:
                outputs = self.forward(inputs)
        return outputs


if __name__ == "__main__":
    # Example usage
    param_file = "/media/PKing/新加卷/SDK/project/cv-sdk-ncnn/cv-sdk-ncnn/data/model/face/pfld98_168.param"

    # Create random input
    batch_size = 1
    num_classes = 4
    input_size = [168, 168]
    inputs = np.random.random(size=(batch_size, 3, input_size[1], input_size[0]))
    inputs = np.asarray(inputs, dtype=np.float32)

    # Create model and run inference
    model = NCNNEngine(param_file, inp_names=['in0'], out_names=['out0'],
                       use_gpu=False, use_fp16=True)
    model.performance(inputs)
    print("----")
