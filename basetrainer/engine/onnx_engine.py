# -*- coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-30 10:40:48
    @Brief  :
"""
import os, sys

sys.path.append(os.getcwd())
import onnx
import onnxruntime
import numpy as np


class ONNXEngine():
    def __init__(self, model_file, use_gpu=False):
        """
        :param model_file:
        :param use_gpu:
        pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
        """
        available_providers = onnxruntime.get_available_providers()
        sess_options = onnxruntime.SessionOptions()
        if use_gpu:
            self.providers = ['CUDAExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(model_file, providers=self.providers,
                                                         sess_options=sess_options)
        self.device = onnxruntime.get_device()
        self.inp_names = self.get_inp_names(self.onnx_session)
        self.out_names = self.get_out_names(self.onnx_session)
        print("available_providers:{},use device:{}".format(available_providers, self.device))
        print("inp_names          :{}".format(self.inp_names))
        print("out_names          :{}".format(self.out_names))
        print("model_file          :{}".format(model_file))
        print('-----------' * 5, flush=True)

    def get_out_names(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_inp_names(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_inp_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

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
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        inp_feed = self.get_inp_feed(self.inp_names, inp_tensor)
        out_tensor = self.onnx_session.run(self.out_names, input_feed=inp_feed)
        return out_tensor

    def performance(self, inputs, iterate=10):
        from pybaseutils import time_utils
        outputs = self.forward(inputs)
        for i in range(iterate):
            with time_utils.Performance() as p:
                outputs = self.forward(inputs)
        return outputs


if __name__ == "__main__":
    model_file = "libs/best.onnx"
    batch_size = 1
    num_classes = 4
    input_size = [320, 320]
    inputs = np.random.random(size=(batch_size, 3, input_size[1], input_size[0]))
    inputs = np.asarray(inputs, dtype=np.float32)
    model = ONNXEngine(model_file, use_gpu=False)
    model.performance(inputs)
    print("----")
