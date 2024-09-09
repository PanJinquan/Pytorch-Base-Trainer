# -*-coding: utf-8 -*-
"""
    @Project: InsightFace_Pytorch
    @File   : evaluation.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-04-30 16:10:41
"""
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import sklearn.model_selection as cross_validation
# import sklearn as sklearn
from sklearn import svm, datasets


def plot_roc_curve(fpr_list, tpr_list, roc_auc_list, line_names):
    '''
    绘制roc曲线
    :param fpr_list:
    :param tpr_list:
    :param roc_auc_list:
    :param line_names:曲线名称
    :return:
    '''
    # 绘图
    # plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    colors = ["b", "r", "c", "m", "g", "lt_steps", "k", "w"]
    for fpr, tpr, roc_auc, color, line_name in zip(fpr_list, tpr_list, roc_auc_list, colors, line_names):
        plt.plot(fpr, tpr, color=color, lw=lw,
                 label='{} ROC curve (area = {:.3f})'.format(line_name, roc_auc))  # 假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')  # 绘制y=1-x的直线

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20,
            }
    plt.xlabel('False Positive Rate', font)
    plt.ylabel('True Positive Rate', font)

    plt.title('ROC curve')
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"

    plt.show()


def get_roc_curve(y_true, y_score, invert=False, plot_roc=True):
    '''
    一般情况下，大于阈值时，y_test为1，小于等于阈值时y_test为0, y_test与y_score一一对应,且是正比关系
    当用距离作为y_score的分数时，此时y_test与y_score是反比关系（大于阈值时，y_test为0，小于等于阈值时y_test为1）
    :param y_true  : 真实值 {0，1}
    :param y_score : 预测分数(0.0~1.0)
    :param invert  : 是否对y_test进行反转，当y_test与y_score是正比关系时，invert=False,当y_test与y_score是反比关系时,invert=True
    :param plot_roc: 是否绘制roc曲线
    :return:fpr,
            tpr,
            roc_auc,
            threshold ,阈值点
            optimal_idx:最佳截断点,best_threshold = threshold[optimal_idx]获得最佳阈值
    '''
    # Compute ROC curve and ROC area for each class
    if invert:
        # y_true = 1 - y_true  # 当y_test与y_score是反比关系时,进行反转
        # or
        y_score=np.max(y_score)-y_score

    # 计算roc
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score, pos_label=1)

    # 计算auc的值
    roc_auc = metrics.auc(fpr, tpr)

    # 计算最优阈值:最佳截断点应该是tpr高,而fpr低的地方。
    # url :https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    optimal_idx = np.argmax(tpr - fpr)
    # best_threshold = threshold[optimal_idx]

    # 绘制ROC曲线
    if plot_roc:
        fpr_list = [fpr]
        tpr_list = [tpr]
        roc_auc_list = [roc_auc]
        line_names = [""]
        plot_roc_curve(fpr_list, tpr_list, roc_auc_list, line_names=line_names)
    # print("fpr:{}".format(fpr))
    # print("tpr:{}".format(tpr))
    # print("threshold:{}".format(threshold))
    # print("roc_auc:{}".format(roc_auc))
    # print("optimal_idx :{},best_threshold :{} ".format(optimal_idx, threshold[optimal_idx]))
    return fpr, tpr, roc_auc, threshold, optimal_idx


def get_tpr_fpr(fpr, tpr, fixed_fpr=0.01):
    """
    metrics_i_string = 'TPR@FPR=10-2: {:.4f}\t'.format(get_tpr_fpr(fpr, tpr,0.01))
    metrics_i_string += 'TPR@FPR=10-3: {:.4f}\t'.format(get_tpr_fpr(fpr, tpr,0.001))
    metrics_i_string += 'TPR@FPR=10-4: {:.4f}\t'.format(get_tpr_fpr(fpr, tpr,0.0001))
    :param fixed_fpr:<float>
    :return:
    """
    # fpr, tpr, thr = metrics.roc_curve(target, output)
    # fpr, tpr, threshold = metrics.roc_curve(y_true, y_score, pos_label=1)
    tpr_filtered = tpr[fpr <= fixed_fpr]
    if len(tpr_filtered) == 0:
        return 0.0
    return tpr_filtered[-1]

def custom_roc_curve(y_true, y_score):
    lw = 2
    plt.figure(figsize=(10, 10))
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    pred_sort = np.sort(y_score)[::-1]  # 从大到小排序
    index = np.argsort(y_score)[::-1]  # 从大到小排序
    y_sort = y_true[index]
    print(y_sort)
    tpr = []
    fpr = []
    thr = []
    for i, item in enumerate(pred_sort):
        tpr.append(np.sum((y_sort[:i] == 1)) / pos)
        fpr.append(np.sum((y_sort[:i] == 0)) / neg)
        thr.append(item)
    print(fpr)
    print(tpr)
    print(thr)
    # 画图
    plt.plot(fpr, tpr, 'k')
    # plt.plot([(0, 0), (1, 1)], 'r--') # lt_steps=x
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')  # 绘制y=1-x的直线

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20,
            }
    plt.xlabel('False Positive Rate', font)
    plt.ylabel('True Positive Rate', font)

    plt.title('ROC curve')
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"

    plt.show()


def save_evaluation(fpr, tpr, roc_auc, save_path):
    np.savez(save_path, fpr=fpr, tpr=tpr, roc_auc=roc_auc)


def load_evaluation(load_path):
    data = np.load(load_path)
    fpr = data['fpr']
    tpr = data['tpr']
    roc_auc = data['roc_auc']
    return fpr, tpr, roc_auc


def iris_roc_test():
    '''
    使用sklearn中的鸢尾花（iris）数据,绘制roc曲线进行测试
    :return:
    '''
    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # 变为2分类
    X, y = X[y != 2], y[y != 2]
    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.3, random_state=0)

    # Learn to predict each class against the other
    svm_model = svm.SVC(kernel='linear', probability=True, random_state=random_state)

    # 通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
    y_score = svm_model.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, roc_auc, threshold, optimal_idx = get_roc_curve(y_true=y_test, y_score=y_score, invert=False,
                                                              plot_roc=True)
    # custom_roc_curve(y_true=y_test, y_score=y_score)
    # print("fpr:{}".format(fpr))
    # print("tpr:{}".format(tpr))
    print("threshold:{}".format(threshold))
    print("roc_auc:{}".format(roc_auc))
    print("optimal_idx :{},best_threshold :{} ".format(optimal_idx, threshold[optimal_idx]))

    # 保存数据
    save_evaluation(fpr, tpr, roc_auc, "evaluation.npz")

    # 加载数据
    fpr, tpr, roc_auc = load_evaluation("evaluation.npz")
    plot_roc_curve(fpr_list=[fpr], tpr_list=[tpr], roc_auc_list=[roc_auc], line_names=["WWW"])


if __name__ == "__main__":
    # 使用sklearn中的鸢尾花（iris）数据,绘制roc曲线进行测试
    iris_roc_test()
    # data_len = 50
    # label = np.random.randint(0, 2, size=data_len)
    # score = np.random.choice(np.arange(0.1, 1, 0.01), data_len)
    # custom_roc_curve(label, y_score=score)