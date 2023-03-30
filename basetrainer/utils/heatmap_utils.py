# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2023-03-28 09:44:09
    @Brief  : Pytorch可视化神经网络热力图(CAM):
              https://blog.csdn.net/sinat_37532065/article/details/103362517
              https://github.com/frgfm/torch-cam
              https://zhuanlan.zhihu.com/p/479485138
"""
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pybaseutils import image_utils, file_utils


class HeatMap(object):
    """绘制类激活映射"""

    def __init__(self, flag="hm", use_grad=True):
        """
        Grad-CAM类激活映射可视化:
        :param flag:
        :param use_grad:
        """
        self.flag = flag
        self.use_grad = use_grad
        self.images = []
        self.names = []
        self.outputs = None
        self.features = None

    def set_image(self, image, name=None):
        if not name: name = f"{file_utils.get_time()}.jpg"
        if not isinstance(image, list): image = [image]
        if not isinstance(name, list): name = [name]
        self.images = image
        self.names = name

    def set_outputs(self, outputs: torch.Tensor):
        self.outputs = outputs

    def set_features(self, features: torch.Tensor):
        self.features = features

    def get_heatmap(self, vis=False, out=""):
        """
        获得热力图可视化结果
        :param vis: 是否显示
        :param out: 是否保存热力图可视化图
        :return: featuremaps： 返回featuremap特征图
                 heatmaps   ： 返回heatmaps热力图
        """
        print(f"features:{self.features.shape}, outputs:{self.outputs.shape}")
        featuremaps, heatmaps = draw_heatmap_from_output(self.outputs, self.features, images=self.images,
                                                         use_grad=self.use_grad, vis=vis)
        if out:
            self.save_heatmap(featuremaps, heatmaps, out)
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


def draw_heatmap_from_output(output: torch.Tensor, features: torch.Tensor, images, use_grad=True, alpha=0.3, vis=False):
    """
    绘制 Class Activation Map
    :param output: 模型输出结果
    :param features: 模型输出特征图
    :param images: 原始图片
    :param use_grad: 是否考虑梯度
    :param vis: 是否可视化原始heatmap（调用matplotlib）
    :return:
    """
    if not isinstance(images, list): images = [images]

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    index = torch.argmax(output).item()
    pred_class = output[:, index].mean()
    if use_grad:
        features.register_hook(extract)
        pred_class.backward()  # 计算梯度
        grads = features_grad  # 获取梯度
        avg_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    featuremaps, heatmaps = [], []
    for b in range(features.size(0)):
        feature = features[b]
        if use_grad:
            avg_grad = avg_grads[b]
            for i in range(feature.size(0)):  feature[i, ...] *= avg_grad[i, ...]
        f, h = plot_heatmap(feature.cpu().detach().numpy(), images[b], alpha=alpha, vis=vis)
        featuremaps.append(f)
        heatmaps.append(h)
    return featuremaps, heatmaps


def draw_heatmap_from_model(model, inputs_tensor, image, use_grad=True, vis=False):
    """
    :param model:
    :param inputs_tensor:
    :param image:
    :param use_grad:
    :param vis:
    :return:
    """
    # 获取模型输出的feature/score
    model.eval()
    scores = model(inputs_tensor)
    features = model.features
    featuremaps, heatmaps = draw_heatmap_from_output(features, scores, images=image, use_grad=use_grad, vis=vis)
    return featuremaps, heatmaps


def plot_heatmap(feature, image, alpha=0.3, vis=False):
    feature = np.mean(feature, axis=0)

    feature = np.maximum(feature, 0)
    feature /= np.max(feature)
    # feature=image_utils.cv_image_normalization(feature,0,1.0)

    # CAM结果保存路径
    H, W = image.shape[:2]
    f = cv2.resize(feature, dsize=(W, H), interpolation=cv2.INTER_LINEAR)  # 将热力图的大小调整为与原始图像相同
    f = np.uint8(np.clip(255 * f, 0, 255))  # 将热力图转换为RGB格式
    f = cv2.applyColorMap(f, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    heatmap = np.float32(f) * (1 - alpha) + np.float32(image) * alpha  # 这里的0.4是热力图强度因子
    heatmap = np.uint8(np.clip(heatmap, 0, 255))  # 将热力图转换为RGB格式

    fig = plt.matshow(feature)
    featuremap = image_utils.fig2data(fig.figure)
    featuremap = cv2.cvtColor(featuremap, cv2.COLOR_RGB2BGR)
    if vis:
        # plt.show()
        image_utils.cv_show_image("featuremap", featuremap, delay=1)
        image_utils.cv_show_image("heatmap", heatmap, delay=10)
    return featuremap, heatmap
