# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: YNet
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-01-06 08:59:23
# --------------------------------------------------------
"""
import numpy as np
import tensorflow as tf
import math


def loss_fun(target, prediction, gamma=1.0, loss_type="l2"):
    if loss_type == "l1":
        loss = l1_loss(target, prediction, gamma)
    elif loss_type == "l2":
        loss = l2_loss(target, prediction, gamma)
    else:
        raise Exception("Error:{}".format(loss_type))
    return loss


def l2_loss(target, prediction, gamma=1.0):
    loss = tf.reduce_mean(tf.square(target - prediction))
    return gamma * loss


def l1_loss(target, prediction, gamma=1.0):
    loss = tf.reduce_mean(tf.abs(target - prediction))
    return gamma * loss


def psnr(target, prediction):
    squares = tf.square(target - prediction, name='squares')
    squares = tf.reshape(squares, [tf.shape(squares)[0], -1])
    # mean psnr over a batch
    p = (-10 / np.log(10)) * tf.compat.v1.disp(tf.reduce_mean(squares, axis=[1]))
    p = tf.reduce_mean(p)
    return p


def psnr_tf(target, prediction):
    p = tf.image.psnr(target, prediction, max_val=1.0)
    p = tf.reduce_mean(p)
    return p


def psnr_keras(y_true, y_pred):
    max_pixel = 1.0
    p = 10.0 * math.log10((max_pixel ** 2) / (tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))))
    return p


def psnr_numpy(im1, im2):
    """
    # im1 和 im2 都为灰度图像，uint8 类型
    :param im1:
    :param im2:
    :return:
    """
    # method 1
    diff = im1 - im2
    mse = np.mean(np.square(diff))
    p = 10 * np.log10(255 * 255 / mse)  # for uint8,[0,255]
    # p = 10 * np.log10(1 * 1 / mse)   # for float32,[0,1]
    return p


def psnr_skimage(im1, im2):
    """
    for uint8,[0,255]
    :param im1:
    :param im2:
    :return:
    """
    import skimage
    # p = skimage.measure.compare_psnr(im1, im2, 255)  # for uint8,[0,255]
    p = skimage.measure.compare_psnr(im1, im2, 255)  # for float32,[0,1]
    return p


if __name__ == "__main__":
    data1 = np.zeros(shape=(1, 100, 100, 3))-0.01
    data2 = np.zeros(shape=(1, 100, 100, 3)) + 0.012
    p1 = psnr_keras(data1, data2)
    p2 = psnr_numpy(data1, data2)
    p3 = psnr_skimage(data1, data2)
    p4 = psnr(data1, data2)
    p5 = psnr_tf(data1, data2)
    print(p1)
    print(p2)
    print(p3)
    print(p4)
    print(p5)
