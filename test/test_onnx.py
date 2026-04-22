"""
This code is used to convert the pytorch model into an onnx format model.
"""
import sys
import os

sys.path.insert(0, os.getcwd())
from easydict import EasyDict
from basetrainer.models import build_models
from basetrainer.utils.converter import pytorch2onnx
from basetrainer.utils import torch_tools


def convert2onnx(model_file, net_type="mobilenet_v2", input_size=[224, 224], num_classes=4, device="cpu"):
    model = build_models.get_models(net_type=net_type,
                                    input_size=input_size,
                                    num_classes=num_classes,
                                    pretrained=False)
    state_dict = torch_tools.load_state_dict(model_file)
    model.load_state_dict(state_dict)
    input_shape = (1, 3, input_size[1], input_size[0])
    onnx_file = model_file.replace(".pth", ".onnx").replace(".pt", ".onnx")
    pytorch2onnx.convert2onnx(model, input_shape, dynamic=True, onnx_file=onnx_file,
                              simplify=True,
                              opset_version=11,
                              device=device)


if __name__ == "__main__":
    model_file = "output/model/best_model_001_40.4686.pth"
    net_type = "mobilenet_v2"
    convert2onnx(model_file, net_type=net_type)