# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-10-17 16:29:11
# @Brief  :
# --------------------------------------------------------
"""

import os
import sys
import MNN
import numpy as np
import cv2
from pybaseutils import time_utils


@time_utils.performance()
def forward(net, session, image):
    # cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)

    # input
    inputTensor = net.getSessionInput(session)
    net.resizeTensor(inputTensor, (1, 3, 224, 224))
    net.resizeSession(session)
    inputTensor.copyFrom(tmp_input)
    # infer
    net.runSession(session)
    outputTensor = net.getSessionOutput(session)
    output_data = outputTensor.getNumpyData()
    # outputTensor.readMap()  # 强制同步点
    return output_data


def modelTest(modelPath):
    # print("Available backends:", MNN.getAvailableBackends())
    net = MNN.Interpreter(modelPath)
    net.setCacheFile(".cachefile")
    # net.setSessionMode(9)
    # net.setSessionHint(0, 20)
    # 调整会话模式
    config = {}
    config['backend'] = "OPENCL"
    # config['backend'] = "CUDA"
    # config['backend'] = "CPU"
    config['precision'] = "high"
    session = net.createSession(config)
    print("Run on backendtype: %d \n" % net.getSessionInfo(session, 2))
    np.random.seed(2020)
    input_shape = (1, 3, 224, 224)
    for i in range(50):
        image = np.random.randn(*input_shape).astype(np.float32)
        output_data = forward(net, session, image)
    print("output=", output_data)


if __name__ == '__main__':
    onnx_file = "data/model/resnet/resnet18_224_224.mnn"
    modelName = onnx_file  # model path
    modelTest(modelName)
