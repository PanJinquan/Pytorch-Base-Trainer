# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-23 14:45:13
    @Brief  :常见问题：
    (1) TRT多模型推理时，Pytorch模型可能会出现冲突，建议Pytorch模型不要使用DataParallel加载模型推理
    (2) 推理输入数据batch_size>1时，需要在转换ONNX模型时，设置dynamic=True
"""
import os
import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from typing import Tuple, List

cuda.init()
device_context = cuda.Device(0).make_context()
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)  # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTEngine(object):
    """TensorRT引擎"""

    def __init__(self, model_file: str, input_shape: tuple):
        """
        https://www.jianshu.com/p/36ff0e224112
        :param model_file: 模型文件，可以是*.trt或者*.onnx文件
        :param input_shape:输入维度(B, C, H, W)
        """
        print("tensorrt:{}".format(trt.__version__))  # 8.4.1.5
        assert len(input_shape) == 4
        B, C, H, W = input_shape
        self.input_size = (W, H)
        self.batch_size = B
        self.input_shape = input_shape
        # Load a serialized engine into memory
        # self.engine = self.load_engine(model_file)  # 加载序列化的cuda引擎
        self.engine = self.build_engine_onnx(model_file=model_file, input_shape=self.input_shape)
        # Create context, this can be re-used  创建 执行环境
        self.context = self.engine.create_execution_context()
        # Profile 0 (first profile) is used by default  context可以设置多个profile， 这里选择第一个，也是默认的profile，其中规定了输入尺寸的变化区间
        self.context.active_optimization_profile = 0
        print("Active Optimization Profile: {}".format(self.context.active_optimization_profile))
        # These binding_idxs can change if either the context or the
        # active_optimization_profile are changed  获得输入输出变量名对应profile的idx
        self.input_binding_idxs, self.output_binding_idxs = self.get_binding_idxs(self.engine,
                                                                                  self.context.active_optimization_profile)
        print("\tinput_shape                    : {}".format(self.input_shape))
        # 获得输入变量的变量名
        self.input_names = [self.engine.get_binding_name(binding_idx) for binding_idx in self.input_binding_idxs]
        print("Input Metadata")
        print("\tNumber of Inputs: {}".format(len(self.input_binding_idxs)))
        print("\tInput names     : {}".format(self.input_names))
        print("\tInput Bindings for Profile {}: {}".format(self.context.active_optimization_profile,
                                                           self.input_binding_idxs))
        print("\tInput names: {}".format(self.input_names))
        # 获得输出变量的变量名
        print("Output Metadata")
        self.output_names = [self.engine.get_binding_name(binding_idx) for binding_idx in self.output_binding_idxs]
        print("\tNumber of Outputs: {}".format(len(self.output_binding_idxs)))
        print("\tOutput names     : {}".format(self.output_names))
        print("\tOutput Bindings for Profile {}: {}\n".format(self.context.active_optimization_profile,
                                                              self.output_binding_idxs), flush=True)
        print("\tTRT model loaded successfully:{}".format(model_file.replace(".onnx", ".trt")))

    @staticmethod
    def build_engine_onnx(model_file: str, input_shape, fp16=True):
        """
        :param model_file: onnx file or trt file
        :param batch_size:
        :param fp16_mode:
        :param max_workspace:
        :return:
        """
        if model_file.endswith("trt") and os.path.exists(model_file):
            trt_file = model_file
            print('Reload engine file: {}'.format(trt_file))
            engine = TRTEngine.load_engine(trt_file)
            return engine
        elif model_file.endswith("onnx"):
            onnx_file = model_file
            trt_file = model_file[:-len("onnx")] + "trt"
        else:
            raise Exception("model_file:{}".format(model_file))
        print("Completed parsing of ONNX file")
        print("convert ONNX file to TRT Engine")
        print("load ONNX  file:{}".format(onnx_file))
        engine = onnx2trt(onnx_file, trt_file=trt_file, input_shape=input_shape, fp16=fp16)
        return engine

    def __call__(self, input_tensor):
        return self.inference(input_tensor)

    def inference(self, image_tensor):
        """
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        :param image_tensor:
        :return:
        """
        input_tensor = [image_tensor]
        outputs = self.forward(input_tensor)
        return outputs

    def forward(self, input_tensor):
        device_context.push()
        # Allocate device memory for inputs. This can be easily re-used if the
        # input shapes don't change  为输入变量赋予host空间，该空间可复用
        device_inputs = [cuda.mem_alloc(h_input.nbytes) for h_input in input_tensor]
        # Copy host inputs to device, this needs to be done for each new input， 由host拷贝到device
        for h_input, d_input in zip(input_tensor, device_inputs):
            cuda.memcpy_htod(d_input, h_input)
        # This needs to be called everytime your input shapes change
        # If your inputs are always the same shape (same batch size, etc.),
        # then you will only need to call this once 重新指定网络输入输出的大小。
        host_outputs, device_outputs = self.setup_binding_shapes(self.engine, self.context, input_tensor,
                                                                 self.input_binding_idxs,
                                                                 self.output_binding_idxs)  # 返回的是输出的idx和device buffer

        # print("\tOutput shapes: {}".format([out.shape for out in host_outputs]))
        # Bindings are a list of device pointers for inputs and outputs
        bindings = device_inputs + device_outputs  # list的合并
        # Inference
        self.context.execute_v2(bindings)
        # Copy outputs back to host to view results 将输出由gpu拷贝到cpu。
        for h_output, d_output in zip(host_outputs, device_outputs):
            cuda.memcpy_dtoh(h_output, d_output)
        # 释放显存
        for b in bindings: b.free()
        device_context.pop()
        return host_outputs

    @staticmethod
    def load_engine(filename: str):
        # Load serialized engine file into memory 加载序列化的cuda引擎并进行反序列化
        with open(filename, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    @staticmethod
    def get_binding_idxs(engine: trt.ICudaEngine, profile_index: int):
        # Calculate start/end binding indices for current context's profile
        num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
        start_binding = profile_index * num_bindings_per_profile
        end_binding = start_binding + num_bindings_per_profile
        print("Engine/Binding Metadata")
        print("\tNumber of optimization profiles: {}".format(engine.num_optimization_profiles))
        print("\tNumber of bindings per profile : {}".format(num_bindings_per_profile))
        print("\tFirst binding for profile {}   : {}".format(profile_index, start_binding))
        print("\tLast binding for profile {}    : {}".format(profile_index, end_binding - 1))

        # Separate input and output binding indices for convenience
        input_binding_idxs = []
        output_binding_idxs = []
        for binding_index in range(start_binding, end_binding):
            if engine.binding_is_input(binding_index):
                input_binding_idxs.append(binding_index)
            else:
                output_binding_idxs.append(binding_index)

        return input_binding_idxs, output_binding_idxs

    @staticmethod
    def setup_binding_shapes(
            engine: trt.ICudaEngine,
            context: trt.IExecutionContext,
            host_inputs: List[np.ndarray],
            input_binding_idxs: List[int],
            output_binding_idxs: List[int],
    ):
        # 指定输入的shape，同时根据输入的shape指定输出的shape，并未输出赋予cuda空间
        # Explicitly set the dynamic input shapes, so the dynamic output
        # shapes can be computed internally
        for host_input, binding_index in zip(host_inputs, input_binding_idxs):
            context.set_binding_shape(binding_index, host_input.shape)
        # assert context.all_binding_shapes_specified
        host_outputs = []
        device_outputs = []
        for binding_index in output_binding_idxs:
            output_shape = context.get_binding_shape(binding_index)
            # Allocate buffers to hold output results after copying back to host
            buffer = np.empty(output_shape, dtype=np.float32)
            host_outputs.append(buffer)
            # Allocate output buffers on device
            device_outputs.append(cuda.mem_alloc(buffer.nbytes))
        return host_outputs, device_outputs


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
    trt_file = "/home/dm/nasdata/release/handwriting/daip-calligraphy-hard/calligraphy-hard-recognizer/app/infercore/resource/resnet18_160_160_c3594_7.onnx"
    # trt_file = "/home/dm/nasdata/release/handwriting/daip-calligraphy-hard/calligraphy-hard-recognizer/app/infercore/resource/resnet18_160_160_c3594_7.trt"
    input_shape = (16, 3, 160, 160)  # (B,C,H,W)
    trt = TRTEngine(trt_file, input_shape=input_shape)
    image = np.ones(shape=input_shape)
    input_tensor = np.asarray(image, dtype=np.float32)
    outputs, = trt(input_tensor)
    print("input_tensor:{}".format(input_tensor.shape))
    print("output      :{}={}".format(outputs.shape, outputs))
