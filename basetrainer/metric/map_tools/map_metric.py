# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :
# @E-mail :
# @Date   : 2020-04-03 18:38:34
# --------------------------------------------------------
"""
import sys
import os

sys.path.append(os.getcwd())
import numpy as np
from tqdm import tqdm
from basetrainer.metric.map_tools.evaluator import map_eval
from pybaseutils import file_utils, json_utils


class MapMetric(object):
    """
    使用方法见evaluate_example或者evaluate_for_json
    mAP参考资料：https://github.com/Cartucho/mAP
    """

    def __init__(self, min_iou=0.5):
        """
        :param min_iou: IOU置信度，大于该值，则认为是TP
        """
        self.min_iou = min_iou

    def evaluate(self, gt_info: dict, dt_info: dict, class_name: list, plot=False):
        """
        :param gt_info: ground-truth results,format is
               gt_info={"class_name1":
                                    {"file-name1":[[score,x1,y1,x2,y2],[score,x1,y1,x2,y2],[]...]}
                                    {"file-name2":[[score,x1,y1,x2,y2],[score,x1,y1,x2,y2],[]...]}，
                        "class_name2":
                                    {"file-name1":[[score,x1,y1,x2,y2],[score,x1,y1,x2,y2],[]...]}
                                    {"file-name2":[[score,x1,y1,x2,y2],[score,x1,y1,x2,y2],[]...]}
                         }
        :param dt_info: detection-results，format is the same with gt_info
        :param class_name: 需要测评的类别
        :param plot: 是否显示绘图结果
        :return:
        """
        ap_logs = "Average Precision Per-class:\n"
        results = {}
        for name in class_name:
            gt = gt_info[name] if name in gt_info else {}
            dt = dt_info[name] if name in dt_info else {}
            if not gt: continue
            if not dt: continue
            r = map_eval.compute_average_precision_per_class(gt, dt, self.min_iou, use_2007_metric=False, name=name)
            results[name] = r
        aps = {k: v["ap"] for k, v in results.items()}
        map = sum(aps.values()) / len(aps)
        ap_logs += "{}\n".format(aps)
        ap_logs += "Average Precision Across All Classes:\nmAP:{:3.5f}".format(map)
        print(ap_logs)
        print("n_classes={}".format(len(class_name)))
        print("==" * 20)
        if plot and map > 0:
            map_eval.plot_map_recall_precision(results)
        return aps, map

    def build_data_info(self, data_info: dict, file_name, boxes: list, label: list, score=[], class_name=[]):
        """
        构建gt_info和dt_info,使用方法见evaluate_example
        :param data_info: 输入/输出结果，用于保存所有目标数据信息
        :param file_name: 用于文件标识的唯一ID，一般用文件名称
        :param boxes: shape is (n,4),is (xmin,ymin,xmax,ymax)
        :param label: shape is (n,),type is np.int32
        :param score: shape is (n,),type is np.float32
        :param class_name:类别列表,用于转换label，class_name[int(label[i])]
        :return:
        """
        if isinstance(label, np.ndarray): label = label.tolist()
        if isinstance(boxes, np.ndarray): boxes = boxes.tolist()
        if isinstance(score, np.ndarray): score = score.tolist()
        for i in range(len(label)):
            n = class_name[int(label[i])] if class_name else label[i]
            b = boxes[i]
            if len(score) > 0:  b = [score[i]] + b
            if n not in data_info: data_info[n] = {}
            if file_name not in data_info[n]: data_info[n][file_name] = []
            data_info[n][file_name].append(b)
        return data_info

    def evaluate_example(self, plot=True):
        """
        使用例子
        :param plot:
        :return:
        """
        class_name = ['BACKGROUND', 'hand']
        dataset = ...
        gt_info = {}
        dt_info = {}
        for i in tqdm(range(len(dataset))):
            file = ...
            # ground-truth results
            gt_boxes = ...  # shape is (n,4),is (xmin,ymin,xmax,ymax)
            gt_label = ...  # shape is (n,),type is np.float32
            self.build_data_info(gt_info, file, gt_boxes, gt_label, class_name=class_name)
            # detection results
            dt_boxes = ...  # shape is (n,4),is (xmin,ymin,xmax,ymax)
            dt_score = ...  # shape is (n,),type is np.float32
            dt_label = ...  # shape is (n,),type is np.int32
            self.build_data_info(dt_info, file, dt_boxes, dt_label, dt_score, class_name=class_name)
        self.evaluate(gt_info, dt_info, class_name, plot=plot)

    def evaluate_for_json(self, dt_file="dt_info.json", gt_file="gt_info.json"):
        """
       dt_file or gt_file format is
        {"class_name1":
                    {"file-name1":[[score,x1,y1,x2,y2],[score,x1,y1,x2,y2],[]...]}
                    {"file-name2":[[score,x1,y1,x2,y2],[score,x1,y1,x2,y2],[]...]}，
        "class_name2":
                    {"file-name1":[[score,x1,y1,x2,y2],[score,x1,y1,x2,y2],[]...]}
                    {"file-name2":[[score,x1,y1,x2,y2],[score,x1,y1,x2,y2],[]...]}
         }
        :param dt_file:
        :param gt_file:
        :return:
        """
        class_name = ['BACKGROUND', 'hand']
        dt_info = json_utils.read_json_data(dt_file)
        gt_info = json_utils.read_json_data(gt_file)
        self.evaluate(gt_info, dt_info, class_name, plot=True)

    def evaluate_for_txt(self, dt_file="", gt_file=""):
        """
        mAP参考资料：https://github.com/Cartucho/mAP
        In these *.txt files, each line should be in the following format:
        class_name1 xmin ymin xmax ymax
        class_name2 xmin ymin xmax ymax
        ...
        :param dt_file:
        :param gt_file:
        :return:
        """
        dt_file = "/home/PKing/nasdata/Project/mAP/input/detection-results"
        gt_file = "/home/PKing/nasdata/Project/mAP/input/ground-truth"
        dt_files = file_utils.get_files_list(dt_file, postfix=["*.txt"])
        gt_files = file_utils.get_files_list(gt_file, postfix=["*.txt"])
        dt_info = {}
        for file in dt_files:
            data_info = file_utils.read_data(file, split=" ")
            name = [info[0] for info in data_info]
            score = [info[1] for info in data_info]
            boxes = [info[2:] for info in data_info]
            self.build_data_info(dt_info, os.path.basename(file), boxes, name, score=score)
        class_name = []
        gt_info = {}
        for file in gt_files:
            data_info = file_utils.read_data(file, split=" ")
            name = [info[0] for info in data_info]
            boxes = [info[1:] for info in data_info]
            class_name += name
            self.build_data_info(gt_info, os.path.basename(file), boxes, name)
        class_name = list(set(class_name))
        class_name = sorted(class_name)
        print(class_name)
        self.evaluate(gt_info, dt_info, class_name, plot=True)


if __name__ == "__main__":
    val = MapMetric()
    val.evaluate_for_json()
    # val.evaluate_for_txt()
