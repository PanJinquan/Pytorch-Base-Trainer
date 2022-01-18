import numpy as np
from sklearn.metrics import roc_curve, accuracy_score
from sklearn import metrics
from . import roc


class AverageMeter(object):
    """计算并存储参数当前值或平均值
    Computes and stores the average and current value
    -------------------------------------------------
    batch_time = AverageMeter()
    即 self = batch_time
    则 batch_time 具有__init__，reset，update三个属性，
    直接使用batch_time.update()调用
    功能为：batch_time.update(time.time() - end)
       仅一个参数，则直接保存参数值
    对应定义：def update(self, val, n=1)
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))
    这些有两个参数则求参数val的均值，保存在avg中
    """

    def __init__(self):
        self.reset()  # __init__():reset parameters

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n == 0:
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MultiAverageMeter(object):
    def __init__(self, keys: list):
        self.keys = keys
        self.avg_meters = {}
        for k in keys: self.avg_meters[k] = AverageMeter()

    def reset(self, keys=None):
        if isinstance(keys, str):
            self.avg_meters[keys].reset()
        elif isinstance(keys, list):
            for k in keys: self.avg_meters[k].reset()
        else:
            for k in self.avg_meters.keys(): self.avg_meters[k].reset()

    def update(self, items: dict, n=1):
        for k, v in items.items(): self.avg_meters[k].update(v, n)


def accuracy(pred, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    :param pred: pred labels,     torch.Size([batch_size, num_classes])
    :param target: target labels, torch.Size([batch_size])
    :param topk: Top K
    :return:
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class ROCMeter(object):
    """Compute TPR with fixed FPR"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.target = np.ones(0)
        self.output = np.ones(0)

    def update(self, output, target):
        """
        Usage:
            test_metrics.update(target.cpu().detach().numpy(), output.cpu().detach().numpy())
            tpr001 = self.test_metrics.get_tpr(0.01)
            metrics_i_string = 'TPR@FPR=10-2: {:.4f}\t'.format(tpr001)
        :param output: <class 'tuple'>: (batch_size, num_classes)
        :param target: <class 'tuple'>: (batch_size,)
        :return:
        """
        # If we use cross-entropy
        if len(output.shape) > 1 and output.shape[1] > 1:
            output = output[:, 1]
        elif len(output.shape) > 1 and output.shape[1] == 1:
            output = output[:, 0]
        self.target = np.hstack([self.target, target])
        self.output = np.hstack([self.output, output])

    def get_tpr(self, fixed_fpr=0.01):
        """
        get TPR
        :param fixed_fpr:<float>
        :return:
        """
        fpr, tpr, thr = roc_curve(self.target, self.output, pos_label=1)
        tpr_filtered = tpr[fpr <= fixed_fpr]
        if len(tpr_filtered) == 0:
            return 0.0
        return tpr_filtered[-1]

    def get_tpr_auc(self, fixed_fpr=0.01, plot_roc=True):
        """
        get TPR and AUC
        :param fixed_fpr:<float>
        :return:
        """
        fpr, tpr, thr = roc_curve(self.target, self.output, pos_label=1)
        tpr_filtered = tpr[fpr <= fixed_fpr]
        # 计算auc的值
        roc_auc = metrics.auc(fpr, tpr)

        # 绘制ROC曲线
        if plot_roc:
            fpr_list = [fpr]
            tpr_list = [tpr]
            roc_auc_list = [roc_auc]
            line_names = [""]
            roc.plot_roc_curve(fpr_list, tpr_list, roc_auc_list, line_names=line_names)

        if len(tpr_filtered) == 0:
            return 0.0, 0.0
        return tpr_filtered[-1], roc_auc

    def get_accuracy(self, thr=0.5):
        """
        get Acc
        :param thr:
        :return:
        """
        acc = accuracy_score(self.target,
                             self.output >= thr)
        return acc

    def get_top_hard_examples(self, top_n=10):
        diff_arr = np.abs(self.target - self.output)
        hard_indexes = np.argsort(diff_arr)[::-1]
        hard_indexes = hard_indexes[:top_n]
        return hard_indexes, self.target[hard_indexes], self.output[hard_indexes]
