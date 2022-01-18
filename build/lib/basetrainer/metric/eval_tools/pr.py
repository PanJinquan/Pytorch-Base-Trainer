# -*-coding: utf-8 -*-
"""
    @Project: utils
    @File   : pr.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-10 17:10:56
"""
# import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import sklearn.model_selection as cross_validation


def get_precision(true_labels, pred_labels, average='binary'):
    '''
    https://blog.csdn.net/CherDW/article/details/55813071
    see @get_precision_recall
    precision=tp / (tp + fp)
    :param true_labels:
    :param pred_labels:
    :return:
    '''
    precision = metrics.precision_score(true_labels, pred_labels, average=average)
    return precision


def get_recall(true_labels, pred_labels, average="binary"):
    '''
    https://blog.csdn.net/CherDW/article/details/55813071
    see @get_precision_recall
    recall=tp / (tp + fn)
    :param true_labels:
    :param pred_labels:
    :param  average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', weighted']
    :return:
    '''
    recall = metrics.recall_score(true_labels, pred_labels, average=average)
    return recall


def get_accuracy(true_labels, pred_labels):
    accuracy = metrics.accuracy_score(true_labels, pred_labels)
    return accuracy


def get_precision_recall_acc(true_labels, pred_labels, average="binary"):
    '''
    将一个二分类matrics拓展到多分类或多标签问题时，我们可以将数据看成多个二分类问题的集合，
    每个类都是一个二分类。接着，我们可以通过跨多个分类计算每个二分类metrics得分的均值，
    这在一些情况下很有用。你可以使用average参数来指定。
    binary :二分类
    macro：计算二分类metrics的均值，为每个类给出相同权重的分值。当小类很重要时会出问题，
          因为该macro-averging方法是对性能的平均。另一方面，该方法假设所有分类都是一样重要的，
          因此macro-averaging方法会对小类的性能影响很大。
    weighted:对于不均衡数量的类来说，计算二分类metrics的平均，通过在每个类的score上进行加权实现。
    micro：给出了每个样本类以及它对整个metrics的贡献的pair（sample-weight），
          而非对整个类的metrics求和，它会每个类的metrics上的权重及因子进行求和，来计算整个份额。
          Micro-averaging方法在多标签（multilabel）问题中设置，包含多分类，此时，大类将被忽略。
    samples：应用在multilabel问题上。它不会计算每个类，相反，它会在评估数据中，
          通过计算真实类和预测类的差异的metrics，来求平均（sample_weight-weighted）
    average：average=None将返回一个数组，它包含了每个类的得分.
    原文：https://blog.csdn.net/CherDW/article/details/55813071
    :param true_labels:
    :param pred_labels:
    :return:
    '''
    precision = get_precision(true_labels, pred_labels, average)
    recall = get_recall(true_labels, pred_labels, average)
    acc = get_accuracy(true_labels, pred_labels)
    return precision, recall, acc


def plot_pr_curve(recall_list, precision_list, auc_list, line_names):
    '''
    绘制roc曲线
    :param recall_list:
    :param precision_list:
    :param auc_list:
    :param line_names:曲线名称
    :return:
    '''
    # 绘图
    # plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    colors = ["b", "r", "c", "m", "g", "lt_steps", "k", "w"]
    for r, p, roc_auc, color, line_name in zip(recall_list, precision_list, auc_list, colors, line_names):
        plt.plot(r, p, color=color, lw=lw,
                 label='{} PR curve (area = {:.3f})'.format(line_name, roc_auc))  # 假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')  # 绘制y=1-x的直线

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20,
            }
    plt.xlabel('Recall', font)
    plt.ylabel('Precision', font)

    plt.title('PR curve')
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"
    plt.show()


def custom_pr_curve(y, pred):
    lw = 2
    plt.figure(figsize=(10, 10))

    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pred_sort = np.sort(pred)[::-1]  # 从大到小排序
    index = np.argsort(pred)[::-1]  # 从大到小排序
    y_sort = y[index]
    print(y_sort)

    Pre = []
    Rec = []
    for i, item in enumerate(pred_sort):
        if i == 0:  # 因为计算precision的时候分母要用到i，当i为0时会出错，所以单独列出
            Pre.append(1)
            Rec.append(0)
        else:
            Pre.append(np.sum((y_sort[:i] == 1)) / i)
            Rec.append(np.sum((y_sort[:i] == 1)) / pos)
    print(Pre)
    print(Rec)
    # 画图
    plt.plot(Rec, Pre, 'k')
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')  # 绘制y=1-x的直线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20,
            }
    plt.xlabel('Recall', font)
    plt.ylabel('Precision', font)

    plt.title('PR curve')
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"
    plt.show()


def get_classification_precision_recall(y_true, probas_pred):
    '''
    对于二分类问题，可以直接调用sklearn.metrics的precision_recall_curve()
    :param y_true: 真实样本的的正负标签
    :param probas_pred: 预测的分数
    precision : array, shape = [n_thresholds + 1]
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.

    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    :return:
    '''

    precision, recall, _thresholds = metrics.precision_recall_curve(y_true, probas_pred)
    AUC = metrics.auc(recall, precision)
    return precision, recall, AUC


def get_object_detection_precision_recall(iou_data, probas_pred, iou_threshold):
    '''
    在二分类中，recall的分母是测试样本的的正样本的个数，因此可认为二分类recall是召回正样本的比率
    sklearn.metrics的precision_recall_curve()是根据二分类返回的recall；
    但对于目标检测问题，召回率recall与二分类的情况不同，目标检测的召回率recall的分母是groundtruth boundingbox的个数
    可以认为目标检测的recall是召回groundtruth boundingbox的比率；
    显示将precision_recall_curve()返回recall*正样本的个数/groundtruth boundingbox个数就是目标检测的recall了
    :param iou_data:
    :param probas_pred:
    :param iou_threshold:
    :return:
    '''
    true_label = np.where(iou_data > iou_threshold, 1, 0)
    t_nums = np.sum(true_label)  # 测试样本中正样本个数
    gb_nums = len(iou_data)  # groundtruth boundingbox的个数
    precision, recall, _thresholds = metrics.precision_recall_curve(true_label, probas_pred)
    recall = recall * t_nums / gb_nums
    AUC = metrics.auc(recall, precision)
    return precision, recall, AUC


def plot_classification_pr_curve(true_label, prob_data, invert=False, plot_pr=True):
    if invert:
        true_label = 1 - true_label  # 当y_test与y_score是反比关系时,进行反转
    precision, recall, AUC = get_classification_precision_recall(y_true=true_label, probas_pred=prob_data)
    # 绘制ROC曲线
    if plot_pr:
        # custom_pr_curve(true_label, prob_data)
        recall_list = [recall]
        precision_list = [precision]
        auc_list = [AUC]
        line_names = ["line_name"]
        plot_pr_curve(recall_list, precision_list, auc_list, line_names)
    return precision, recall, AUC


def plot_object_detection_pr_curve(iou_data, prob_data, iou_threshold, plot_pr=True):
    precision, recall, AUC = get_object_detection_precision_recall(iou_data=iou_data, probas_pred=prob_data,
                                                                   iou_threshold=iou_threshold)
    # 绘制ROC曲线
    if plot_pr:
        recall_list = [recall]
        precision_list = [precision]
        auc_list = [AUC]
        line_names = [""]
        plot_pr_curve(recall_list, precision_list, auc_list, line_names)


def precision_recall_demo():
    true_labels = [0, 1, 2, 3, 4, 5]
    pred_labels = [0, 2, 1, 0, 0, 1]
    # true_labels = [0, 1, 3, 2, 1]
    # pred_labels = [0, 1, 1, 0, 0]
    precision, recall, acc = get_precision_recall_acc(true_labels, pred_labels, average="macro")
    print("precision:{},recall:{},acc:{}".format(precision, recall, acc))


if __name__ == "__main__":
    iou_data = [0.88, 0.4, 0.70]
    prob_data = [0.9, 0.8, 0.7]
    iou_data = np.array(iou_data)
    prob_data = np.array(prob_data)
    # 阈值
    iou_threshold = 0.5
    true_label = np.where(iou_data > iou_threshold, 1, 0)
    # plot_classification_pr_curve(true_label, prob_data)
    # plot_object_detection_pr_curve(iou_data, prob_data, iou_threshold, plot_pr=True)

    precision_recall_demo()
