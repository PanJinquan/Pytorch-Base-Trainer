# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-23 11:06:28
    @Brief  : https://blog.csdn.net/weixin_42492254/article/details/125319112
"""

import os
import tensorrt as trt
import sys

print("tensorrt:{}".format(trt.__version__)) # 8.4.3.1

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger()


def ONNX2RTR(onnx_file, trt_file="", input_shape=(1, 3, 160, 160), max_batch_size=8, FP16=True, set_shape=True):
    """
    Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it.
    :param onnx_file:
    :param trt_file:
    :return:
    """
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    BATCH_SIZE, C, H, W = input_shape
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser, \
            trt.Runtime(TRT_LOGGER) as runtime:
        # Parse model file
        if not os.path.exists(onnx_file):
            print("ONNX file {} not found".format(onnx_file))
            exit(0)
        print("Loading ONNX file from path {}...".format(onnx_file))
        with open(onnx_file, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print("Completed parsing of ONNX file")
        print("Building an engine from file {}; this may take a while...".format(onnx_file))
        builder.max_batch_size = max_batch_size
        config = builder.create_builder_config()
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(2))
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1 << 30)
        config.max_workspace_size = 1 << 30
        if FP16: config.set_flag(trt.BuilderFlag.FP16)
        # if INT8:config.set_flag(trt.BuilderFlag.INT8)
        if set_shape:
            profile = builder.create_optimization_profile()  # 动态输入时候需要 分别为最小输入、常规输入、最大输入
            # 有几个输入就要写几个profile.set_shape 名字和转onnx的时候要对应
            # tensorrt6以后的版本是支持动态输入的，需要给每个动态输入绑定一个profile，用于指定最小值，常规值和最大值，如果超出这个范围会报异常。
            profile.set_shape_input("input", (1, C, H, W), (1, C, H, W), (max_batch_size, C, H, W))
            # profile.set_shape_input('input', *[[BATCH_SIZE, 3, H, W]] * 3)
            config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        print("Completed creating Engine")
        with open(trt_file, "wb") as f:
            f.write(serialized_engine)

        # engine = builder.build_engine(network, config)
        # with open('model_python_trt.engine', mode='wb') as f:
        #     f.write(bytearray(engine.serialize()))
        #     print("generating file done!")
    print("input_shape=({}), max_batch_size={}".format(input_shape, max_batch_size))
    print("output engine_file:{}".format(trt_file))
    return engine


if __name__ == "__main__":
    onnx_file = '/home/dm/nasdata/dataset-dmai/handwriting/word-class/model/书法文字识别-V4.0(3594字+7)/resnet18_1.0_160_160_LabelSmooth_SGD_20221209095813/model/best_model_119_99.7698_sim.onnx'
    trt_file = "/home/dm/nasdata/dataset-dmai/handwriting/word-class/model/书法文字识别-V4.0(3594字+7)/resnet18_1.0_160_160_LabelSmooth_SGD_20221209095813/model/best_model_119_99.7698_sim.trt"
    ONNX2RTR(onnx_file, trt_file, max_batch_size=1)
