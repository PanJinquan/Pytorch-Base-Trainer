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

print("tensorrt:{}".format(trt.__version__))  # 8.4.3.1

TRT_LOGGER = trt.Logger()


def onnx2trt(onnx_file, trt_file=None, input_shape=(1, 3, 160, 160), max_batch_size=8, fp16=True):
    """
    :param onnx_file: ONNX模型文件
    :param trt_file: 输出TRT模型文件，默认为onnx_file同目录
    :param input_shape: 模型输入维度
    :param fp16: 是否进行半精度量化
    :return:
    """

    def GiB(val):
        return val * 1 << 30

    if not trt_file: trt_file = onnx_file[:-len("onnx")] + "trt"
    batch_size, C, H, W = input_shape
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = max_batch_size  # trt推理时最大支持的batchsize
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(4)
        if fp16: config.set_flag(trt.BuilderFlag.FP16)
        # if int8:config.set_flag(trt.BuilderFlag.INT8)
        print('Loading ONNX file from path {}...'.format(onnx_file))
        with open(onnx_file, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file))
        profile = builder.create_optimization_profile()  # 动态输入时候需要 分别为最小输入、常规输入、最大输入
        # 有几个输入就要写几个profile.set_shape 名字和转onnx的时候要对应
        # tensorrt6以后的版本是支持动态输入的，需要给每个动态输入绑定一个profile，用于指定最小值，常规值和最大值，如果超出这个范围会报异常。
        profile.set_shape("input", (1, C, H, W), (batch_size, C, H, W), (batch_size * 2, C, H, W))
        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config)
    print("Completed creating Engine")
    # 保存engine文件
    with open(trt_file, "wb") as f:
        f.write(engine.serialize())
    print("save TRT file:{}".format(trt_file))
    return engine


if __name__ == "__main__":
    onnx_file = '/home/dm/nasdata/release/handwriting/daip-calligraphy-hard/calligraphy-hard-recognizer/app/infercore/resource/resnet18_160_160_c3594_7.onnx'
    onnx2trt(onnx_file, input_shape=(1, 3, 160, 160))
