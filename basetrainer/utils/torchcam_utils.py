# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2023-03-29 20:09:27
    @Brief  : Pytorch可视化神经网络热力图(CAM):
              https://blog.csdn.net/sinat_37532065/article/details/103362517
              https://github.com/frgfm/torch-cam
              https://zhuanlan.zhihu.com/p/479485138
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import torchcam
from pybaseutils import file_utils, image_utils
from torchcam import methods

CAM_TYPE = ["CAM", "GradCAM", "GradCAMpp", "SmoothGradCAMpp", "XGradCAM", "LayerCAM"]


class TorchCAM(object):
    """绘制类激活映射"""

    def __init__(self, model: torch.nn.Module, target_layer=None, input_shape=(3, 224, 224),
                 cam_type="GradCAM", flag="hm"):
        """
        类激活映射: https://github.com/frgfm/torch-cam
        :param model: 模型
        :param target_layer: 可视化指定层的名称
        :param input_shape: 模型输入维度(C,H,W)
        :param cam_type: 可视化方法： ["CAM", "GradCAM", "GradCAMpp", "SmoothGradCAMpp", "XGradCAM", "LayerCAM"]
        :param flag:
        """
        self.flag = flag
        self.images = []
        self.names = []
        self.outputs = None
        self.features = None
        self.model = model
        if len(input_shape) == 2:
            input_shape = (3, input_shape[1], input_shape[0])
        elif len(input_shape) == 4:
            b, c, h, w = input_shape
            input_shape = (c, h, w)
        self.cam = self.build_cam(cam_type=cam_type, target_layer=target_layer, input_shape=input_shape)
        self.target_layer = self.cam.target_names
        if target_layer is None:
            print(f"target_layer is not specified, automatic resolution target_layer is {self.target_layer}")
        else:
            print(f"target_layer is {self.target_layer}")

    def build_cam(self, cam_type="GradCAM", target_layer=None, input_shape=(3, 224, 224)):
        """
        :param cam_type: ["GradCAM", "GradCAMpp", "SmoothGradCAMpp", "XGradCAM", "LayerCAM"]
        :param target_layer:
        :param input_shape:
        :return:
        """
        if cam_type.lower() == "CAM".lower():
            cam = methods.CAM(self.model, target_layer=target_layer, input_shape=input_shape)
        elif cam_type.lower() == "GradCAM".lower():
            cam = methods.GradCAM(self.model, target_layer=target_layer, input_shape=input_shape)
        elif cam_type.lower() == "GradCAMpp".lower():
            cam = methods.GradCAMpp(self.model, target_layer=target_layer, input_shape=input_shape)
        elif cam_type.lower() == "SmoothGradCAMpp".lower():
            cam = methods.SmoothGradCAMpp(self.model, target_layer=target_layer, input_shape=input_shape)
        elif cam_type.lower() == "XGradCAM".lower():
            cam = methods.XGradCAM(self.model, target_layer=target_layer, input_shape=input_shape)
        elif cam_type.lower() == "LayerCAM".lower():
            cam = methods.LayerCAM(self.model, target_layer=target_layer, input_shape=input_shape)
        else:
            cam = None
        return cam

    def set_image(self, image, name=None):
        if not name: name = f"{file_utils.get_time()}.jpg"
        if not isinstance(image, list): image = [image]
        if not isinstance(name, list): name = [name]
        self.images = image
        self.names = name

    def set_outputs(self, outputs: torch.Tensor):
        self.outputs = outputs

    def set_features(self, features: torch.Tensor):
        # self.features = features
        pass

    def get_heatmap(self, vis=False, out=""):
        """
        获得热力图可视化结果
        :param vis: 是否显示
        :param out: 是否保存热力图可视化图
        :return: featuremaps： 返回featuremap特征图
                 heatmaps   ： 返回heatmaps热力图
        """
        pred_index = self.outputs.argmax().item()
        outputs = self.cam(class_idx=pred_index, scores=self.outputs)
        features, = outputs
        featuremaps, heatmaps = self.draw_heatmap_from_output(features, vis=vis)
        if out:
            self.save_heatmap(featuremaps, heatmaps, out)
        return featuremaps, heatmaps

    def draw_heatmap_from_output(self, features: torch.Tensor, vis=False):
        heatmaps = []
        featuremaps = []
        features = features.cpu().detach().numpy()
        for b in range(len(features)):
            feature = features[b]
            mask = np.array(feature, dtype=np.float32)
            image = Image.fromarray(self.images[b])
            mask = Image.fromarray(mask)
            heatmap = torchcam.utils.overlay_mask(image, mask, alpha=0.3)
            heatmap = np.array(heatmap)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            #
            feature = np.mean(feature, axis=0)
            feature = np.maximum(feature, 0)
            feature /= np.max(feature)
            fig = plt.matshow(features[b])
            featuremap = image_utils.fig2data(fig.figure)
            featuremap = cv2.cvtColor(featuremap, cv2.COLOR_RGB2BGR)

            featuremaps.append(featuremap)
            heatmaps.append(heatmap)
            if vis:
                image_utils.cv_show_image("featuremap", featuremap, delay=1)
                image_utils.cv_show_image("heatmap", heatmap, delay=10)
        return featuremaps, heatmaps

    def save_heatmap(self, featuremaps, heatmaps, out: str, combine=True):
        if not out: return
        for i in range(len(featuremaps)):
            name = os.path.basename(self.names[i]).split(".")[0]
            if combine:
                file = file_utils.create_dir(out, None, f"{name}-{self.flag}.jpg")
                image = image_utils.image_hstack([featuremaps[i], heatmaps[i]])
                cv2.imwrite(file, image)
            else:
                hfile = file_utils.create_dir(out, None, f"{name}-{self.flag}-featuremap.jpg")
                vfile = file_utils.create_dir(out, None, f"{name}-{self.flag}-heatmap-image.jpg")
                cv2.imwrite(hfile, featuremaps[i])
                cv2.imwrite(vfile, heatmaps[i])
