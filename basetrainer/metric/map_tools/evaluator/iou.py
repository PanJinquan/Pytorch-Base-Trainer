# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : iou.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-10 10:14:56
"""

import numpy as np


def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou


def cal_iou_list(box1, box2_list):
    iou_list = []
    for box2 in box2_list:
        iou = cal_iou(box1, box2)
        iou_list.append(iou)
    return iou_list


def get_iou_mat(box1_list, box2_list):
    '''
    获得IOU矩阵:
    :param box1_list: len1
    :param box2_list: len2
    :return:iou_mat = <np.ndarray>: (len1, len2)
    '''
    iou_mat = []
    for box1 in box1_list:
        iou_list = cal_iou_list(box1, box2_list)
        # iou_list = cal_bbox_distance_list(box1, box2_list)
        iou_mat.append(iou_list)
    iou_mat = np.asarray(iou_mat)
    return iou_mat


def get_max_iou_index(iou_mat, iou_threshold):
    max_index = np.argmax(iou_mat, axis=1)
    max_iou = np.max(iou_mat, axis=1)
    index = max_iou > iou_threshold
    max_iou = max_iou[index]
    max_index = max_index[index]
    return max_iou, max_index


def cal_distance(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    center1 = np.asarray([xmin1 + xmax1, ymin1 + ymax1]) / 2.0
    center2 = np.asarray([xmin2 + xmax2, ymin2 + ymax2]) / 2.0
    d = np.sqrt(np.sum(np.power((center1 - center2), 2)))
    return d


def cal_bbox_distance_list(box1, box2_list):
    dist_list = []
    for box2 in box2_list:
        d = cal_distance(box1, box2)
        dist_list.append(d)
    return dist_list


if __name__ == '__main__':
    box1 = [661, 27, 679, 47]
    box2 = [661, 27, 679, 47]
    box2_list = []
    box2_list.append(box2)
    box2_list.append(box2)
    iou = cal_iou_list(box1, box2_list)
    print(iou)

if __name__ == '__main__':
    box1 = [661, 27, 679, 47]
    box2 = [661, 27, 679, 47]
    box2_list = []
    box2_list.append(box2)
    box2_list.append(box2)
    iou = cal_iou_list(box1, box2_list)
    print(iou)
