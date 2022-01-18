# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : acc.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-12 18:22:29
"""

import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np


def plot_acc_curve(acc_list, threshold_list, line_names,title='Accuracy-Threshold'):
    '''
    绘制roc曲线
    :param acc_list:
    :param threshold_list:
    :param roc_auc_list:
    :param line_names:曲线名称
    :return:
    '''
    # 绘图
    # plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    colors = ["b", "r", "c", "m", "g", "lt_steps", "k", "w"]
    xlim_max = 0
    for acc, th, color, line_name in zip(acc_list, threshold_list, colors, line_names):
        max_acc = max(acc)
        if xlim_max < max(th):
            xlim_max = max(th)
        plt.plot(th, acc, color=color, lw=lw,
                 label='{} max Accuracy:{:.3f})'.format(line_name, max_acc))  # 假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')  # 绘制y=1-x的直线

    plt.xlim([0.0, xlim_max])

    plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20,
            }
    plt.xlabel('Threshold', font)
    plt.ylabel('Accuracy ', font)

    plt.title(title)
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"

    plt.show()


def get_accuracy_list(y_true, y_pred, threshold_list, invert=False, plot_acc=True):
    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred)
    if isinstance(y_true, list):
        y_true = np.asarray(y_true)

    acc_list = []
    for th in threshold_list:
        if invert:
            pred_label = np.where(y_pred <= th, 1, 0)
            # pred_label = np.less(y_pred, th)
        else:
            pred_label = np.where(y_pred >= th, 1, 0)
            # pred_label = np.greater(y_pred, th)

        true_label = y_true
        accuracy = metrics.accuracy_score(true_label, pred_label)
        acc_list.append(accuracy)
    max_acc = max(acc_list)
    index = np.where(np.asarray(acc_list) == max_acc)[0]
    best_acc_index = max(index)
    best_threshold = threshold_list[best_acc_index]
    # print("acc_list    :{}".format(acc_list))
    # print("max accuracy:{},best_acc_index:>{},best_threshold:>{}".format(max_acc, best_acc_index,best_threshold))
    if plot_acc:
        acc_list = [acc_list]
        threshold_list = [threshold_list]
        line_names = [""]
        title = 'Accuracy-Threshold,BT:{}'.format(best_threshold)
        plot_acc_curve(acc_list, threshold_list, line_names,title=title)
    return max_acc,best_threshold


if __name__ == "__main__":
    y_pred = [0, 0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2, 0.0]
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    get_accuracy_list(y_true, y_pred, threshold_list=[0.1, 0.2, 0.4, 0.5])
