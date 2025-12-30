# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : PKing
# @E-mail : pan_jinquan@163.com
# @Date   : 2024-02-18 16:10:49
# @Brief  : 常见问题
    (1) TRT多模型推理时，Pytorch模型可能会出现冲突，建议Pytorch模型不要使用DataParallel加载模型推理
    (2) 推理输入数据batch_size>1时，需要在转换ONNX模型时，设置dynamic=True
    (3) 多次`import pycuda.autoinit` 可能出现异常:Error Code 1: Cask (Cask convolution execution)
    (4) 若出现cuDNN error: CUDNN_STATUS_MAPPING_ERROR错误，尝试设置
         torch.cuda.set_device('cuda:0') # fix error: CUDNN_STATUS_MAPPING_ERROR错误
         # torch.backends.cudnn.enabled = False
    (5) Pytorch使用半精度进行模型推理时，需要执行torch.cuda.empty_cache()，显存才会显著下降，否则显存比全精度还高
         torch.cuda.empty_cache() # 清空显存缓存
     https://blog.csdn.net/weixin_44613415/article/details/131850160
# --------------------------------------------------------
"""
import os
import cv2
import numpy as np
import os
import tensorrt as trt
import pycuda.driver as cuda
import ctypes
import logging
from pybaseutils import file_utils, image_utils

logger = logging.getLogger(__name__)
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def build_transform(image, input_size):
    """
    数据预处理操作
    :param image:
    :param input_size:
    :return:
    """
    image = cv2.resize(image, (input_size[1], input_size[0]))
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image /= 255.0
    return image


class ImageDataset(object):
    """数据加载类用于加载校准数据"""

    def __init__(self, image_dir, transform=build_transform, input_size=(640, 640), use_rgb=True, batch_size=8,
                 epoch=10):
        """
        :param image_dir:
        :param transform:
        :param input_size:
        :param use_rgb:
        :param batch_size:
        :param epoch:
        """
        self.index = 0
        self.epoch = epoch
        self.use_rgb = use_rgb
        self.input_size = input_size
        self.batch_size = batch_size
        self.transform = transform
        self.file_list = file_utils.get_images_list(image_dir)
        self.nums = len(self.file_list)  # 图片个数
        assert self.nums > 0, Exception("图片为空：{}".format(image_dir))
        self.iteration = self.batch_size * self.epoch  # 总迭代数
        if self.nums < self.iteration:
            self.file_list = self.file_list * ((self.iteration + self.nums - 1) // self.nums)
        self.batch_data = np.zeros((self.batch_size, 3, self.input_size[1], self.input_size[0]), dtype=np.float32)
        print("images:{}/{},total iteration:{}".format(self.nums, len(self.file_list), self.iteration))
        print("epoch:{},batch_size:{},input_size :{}".format(self.epoch, self.batch_size, self.input_size))

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.epoch:
            for i in range(self.batch_size):
                image_file = self.file_list[i + self.index * self.batch_size]
                image = cv2.imread(image_file)
                if self.use_rgb: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.transform: image = self.transform(image, input_size=self.input_size)
                self.batch_data[i] = image
            self.index += 1
            return np.ascontiguousarray(self.batch_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.epoch


class Calibrator(trt.IInt8EntropyCalibrator):
    """
    校准器Calibrator,TensorRT提供的四个校准器类中的一个，需要重写父校准器的几个方法
    IInt8EntropyCalibrator2
    IInt8LegacyCalibrator
    IInt8EntropyCalibrator
    IInt8MinMaxCalibrator
    """

    def __init__(self, dataset: ImageDataset, cache_file="./calibration.cache"):
        """
        :param dataset:
        :param cache_file:
        """
        trt.IInt8EntropyCalibrator.__init__(self)
        self.dataset = dataset
        self.d_input = cuda.mem_alloc(self.dataset.batch_data.nbytes)
        self.cache_file = cache_file
        dataset.reset()

    def get_batch_size(self):
        """用于获取batch的大小"""
        return self.dataset.batch_size

    def get_batch(self, names):
        """用于获取一个batch的数据"""
        batch = self.dataset.next_batch()
        if not batch.size:
            return None
        # 把校准数据从CPU搬运到GPU中
        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        """用于从文件中读取校准表"""
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """用于把校准表从内存中写入文件中"""
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)


if __name__ == '__main__':
    image_dir = "/home/PKing/nasdata/Project/3D/Depth-Anything/Depth-Anything/assets/test_image"
    dataset = ImageDataset(image_dir, input_size=(640, 640), use_rgb=True, batch_size=8, epoch=2)
    for i in range(dataset.epoch):
        data = dataset.next_batch()
        print(data.shape)
