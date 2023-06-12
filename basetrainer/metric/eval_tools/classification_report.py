# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : classification_report.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-13 13:23:05
"""
import os
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import pandas_tools


def plot_confusion_matrix(conf_matrix, labels_name, title, normalization=True):
    if normalization:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(conf_matrix, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_confusion_matrix(true_labels, pred_labels, target_names=None, filename=None, normalization=True, plot=False,
                         title="Confusion Matrix"):
    '''
    :param true_labels: Y-ylabel
    :param pred_labels: X-xlabel
    :param target_names: 如果输入的true_labels和pred_labels都是Int类型的label,
                         则可以使用target_names映射为class_name
    :param filename: 保存混淆矩阵文件
    :param normalization: 是将混淆矩阵转换为概率形式
    :param plot: 是否绘制混淆矩阵
    :param title:
    :return:
    '''
    if target_names is None:
        target_names = list(set(pred_labels) | set(true_labels))
        target_names.sort()
    else:
        true_labels = [target_names[int(i)] for i in true_labels]
        pred_labels = [target_names[int(i)] for i in pred_labels]
    conf_matrix = metrics.confusion_matrix(true_labels, pred_labels, labels=target_names)
    if normalization:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # 归一化
    pdf = pd.DataFrame(conf_matrix, columns=target_names, index=target_names)
    # print("Confusion Matrix:\n",pdf)
    if filename is not None:
        create_file_path(filename)
        pandas_tools.save_csv(filename, pdf, save_index=True)
    if plot:
        plot_confusion_matrix(conf_matrix, target_names, title, normalization=normalization)
    return conf_matrix


def get_classification_report(true_labels, pred_labels, labels=None, target_names=None, output_dict=False):
    """
    true_labels = [0, 1, 2, 3, 4, 1]  # Y
    pred_labels = [0, 1, 1, 2, 2, 1]  # X
    target_names = ["A", "B", "C", "D", "E"]
    out_result = get_classification_report(true_labels, pred_labels, target_names=target_names, output_dict=False)
    宏平均(macro avg)和微平均(micro avg)
    如果每个class的样本数量差不多,那么宏平均和微平均没有太大差异
    如果每个class的样本数量差异很大,而且你想:
    更注重样本量多的class:使用微平均,若微平均比宏平均小,应检检查样本量多的class
    更注重样本量少的class:使用宏平均,若宏平均比微平均小,应检查样本量少的class
    :param true_labels:
    :param pred_labels:
    :param labels:
    :param target_names:
    :param output_dict:
    :return:
    """
    true_labels = np.array(true_labels, dtype=np.int32)
    pred_labels = np.array(pred_labels, dtype=np.int32)
    if target_names:
        labels = list(range(len(target_names)))
    result = metrics.classification_report(true_labels,
                                           pred_labels,
                                           labels=labels,
                                           digits=4,
                                           target_names=target_names,
                                           output_dict=output_dict)
    if output_dict:
        macro_avg = result["macro avg"]
        accuracy = result["accuracy"]
        weighted_avg = result["weighted avg"]
        out_result = {"macro_avg": macro_avg, "accuracy": accuracy, "weighted_avg": weighted_avg}
        # pdf=pd.DataFrame.from_dict(result)
        # save_csv("classification_report.csv", pdf)

    else:
        out_result = result
    return out_result


def create_dir(parent_dir, dir1=None, filename=None):
    """
    create directory
    :param parent_dir:
    :param dir1:
    :param filename:
    :return:
    """
    out_path = parent_dir
    if dir1:
        out_path = os.path.join(parent_dir, dir1)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if filename:
        out_path = os.path.join(out_path, filename)
    return out_path


def create_file_path(filename):
    """
    create file in path
    :param filename:
    :return:
    """
    basename = os.path.basename(filename)
    dirname = os.path.dirname(filename)
    out_path = create_dir(dirname, dir1=None, filename=basename)
    return out_path


if __name__ == "__main__":
    true_labels = [0, 1, 2, 3, 3, 1, 1]  # Y
    pred_labels = [0, 1, 2, 2, 2, 1, 0]  # X
    # true_labels = [0, 1, 1, 2, 2]
    # pred_labels = [0, 1, 1, 2, 2]
    target_names = ["A", "B", "C", "D"]
    out_result = get_classification_report(true_labels, pred_labels, target_names=target_names, output_dict=False)
    print(out_result)
    # get_confusion_matrix(true_labels, pred_labels, class_names=None, normalization=False,plot=True, title="NVR Confusion Matrix")
