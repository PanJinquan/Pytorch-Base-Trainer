import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.convbn2d_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(5,5), out_channels=24, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.feature_2 = nn.ReLU()
        self.convbn2d_1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=24, kernel_size=(3,3), out_channels=24, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.feature_5 = nn.ReLU()
        self.convbn2d_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=24, kernel_size=(3,3), out_channels=48, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.feature_8 = nn.ReLU()
        self.convbn2d_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=48, kernel_size=(3,3), out_channels=48, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.feature_11 = nn.ReLU()
        self.feature_12 = nn.MaxPool2d(ceil_mode=False, dilation=(1,1), kernel_size=(3,3), padding=(1,1), return_indices=False, stride=(2,2))
        self.convbn2d_4 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=48, kernel_size=(3,3), out_channels=96, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.feature_15 = nn.ReLU()
        self.convbn2d_5 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=96, kernel_size=(3,3), out_channels=96, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.feature_18 = nn.ReLU()
        self.feature_19 = nn.MaxPool2d(ceil_mode=False, dilation=(1,1), kernel_size=(3,3), padding=(1,1), return_indices=False, stride=(2,2))
        self.convbn2d_6 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=96, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.feature_22 = nn.ReLU()
        self.convbn2d_7 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.feature_25 = nn.ReLU()
        self.feature_26 = nn.MaxPool2d(ceil_mode=False, dilation=(1,1), kernel_size=(3,3), padding=(0,0), return_indices=False, stride=(2,2))
        self.convbn2d_8 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=192, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.feature_29 = nn.ReLU()
        self.convbn2d_9 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=192, kernel_size=(3,3), out_channels=192, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.feature_32 = nn.ReLU()
        self.loc = nn.MaxPool2d(ceil_mode=False, dilation=(1,1), kernel_size=(5,2), padding=(0,1), return_indices=False, stride=(1,1))
        self.newCnn = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=192, kernel_size=(1,1), out_channels=23, padding=(0,0), padding_mode='zeros', stride=(1,1))

        archive = zipfile.ZipFile('./output/model.pnnx.bin', 'r')
        self.convbn2d_0.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_0.bias', (24), 'float32')
        self.convbn2d_0.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_0.weight', (24,3,5,5), 'float32')
        self.convbn2d_1.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_1.bias', (24), 'float32')
        self.convbn2d_1.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_1.weight', (24,24,3,3), 'float32')
        self.convbn2d_2.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_2.bias', (48), 'float32')
        self.convbn2d_2.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_2.weight', (48,24,3,3), 'float32')
        self.convbn2d_3.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_3.bias', (48), 'float32')
        self.convbn2d_3.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_3.weight', (48,48,3,3), 'float32')
        self.convbn2d_4.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_4.bias', (96), 'float32')
        self.convbn2d_4.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_4.weight', (96,48,3,3), 'float32')
        self.convbn2d_5.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_5.bias', (96), 'float32')
        self.convbn2d_5.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_5.weight', (96,96,3,3), 'float32')
        self.convbn2d_6.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_6.bias', (128), 'float32')
        self.convbn2d_6.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_6.weight', (128,96,3,3), 'float32')
        self.convbn2d_7.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_7.bias', (128), 'float32')
        self.convbn2d_7.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_7.weight', (128,128,3,3), 'float32')
        self.convbn2d_8.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_8.bias', (192), 'float32')
        self.convbn2d_8.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_8.weight', (192,128,3,3), 'float32')
        self.convbn2d_9.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_9.bias', (192), 'float32')
        self.convbn2d_9.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_9.weight', (192,192,3,3), 'float32')
        self.newCnn.bias = self.load_pnnx_bin_as_parameter(archive, 'newCnn.bias', (23), 'float32')
        self.newCnn.weight = self.load_pnnx_bin_as_parameter(archive, 'newCnn.weight', (23,192,1,1), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.convbn2d_0(v_0)
        v_2 = self.feature_2(v_1)
        v_3 = self.convbn2d_1(v_2)
        v_4 = self.feature_5(v_3)
        v_5 = self.convbn2d_2(v_4)
        v_6 = self.feature_8(v_5)
        v_7 = self.convbn2d_3(v_6)
        v_8 = self.feature_11(v_7)
        v_9 = self.feature_12(v_8)
        v_10 = self.convbn2d_4(v_9)
        v_11 = self.feature_15(v_10)
        v_12 = self.convbn2d_5(v_11)
        v_13 = self.feature_18(v_12)
        v_14 = self.feature_19(v_13)
        v_15 = self.convbn2d_6(v_14)
        v_16 = self.feature_22(v_15)
        v_17 = self.convbn2d_7(v_16)
        v_18 = self.feature_25(v_17)
        v_19 = self.feature_26(v_18)
        v_20 = self.convbn2d_8(v_19)
        v_21 = self.feature_29(v_20)
        v_22 = self.convbn2d_9(v_21)
        v_23 = self.feature_32(v_22)
        v_24 = self.loc(v_23)
        v_25 = self.newCnn(v_24)
        v_26 = torch.squeeze(input=v_25, dim=2)
        v_27 = torch.transpose(input=v_26, dim0=2, dim1=1)
        return v_27

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 48, 256, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("./output/model_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 48, 256, dtype=torch.float)

    torch.onnx.export(net, v_0, "./output/model_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 48, 256, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
