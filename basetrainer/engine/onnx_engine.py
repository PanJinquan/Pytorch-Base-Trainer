# -*-coding: utf-8 -*-

import os, sys

sys.path.append(os.getcwd())
import onnxruntime
import onnx


class ONNXEngine():
    def __init__(self, onnx_path, use_gpu=False):
        """
        :param onnx_path:
        :param use_gpu:
        pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
        """
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['TensorrtExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
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

    def forward(self, image_tensor):
        """
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        """
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        outputs = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return outputs


if __name__ == "__main__":
    import numpy as np

    model_file = ""
    prune_file = ""
    batch_size = 1
    num_classes = 4
    input_size = [128, 128]
    inputs = np.random.random(size=(batch_size, 3, input_size[1], input_size[0]))
    inputs = np.asarray(inputs, dtype=np.float32)
    model = ONNXEngine(model_file)
    print("----")
