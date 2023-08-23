# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-03 18:38:34
# --------------------------------------------------------
"""
import sys
import os

sys.path.append(os.getcwd())
from basetrainer.metric.map_tools.evaluator import measurements
from pybaseutils import file_utils, json_utils, base64_utils


class MapMetric(object):
    def __init__(self, iou_threshold=0.5, prob_threshold=0.05):
        """
        :param iou_threshold: IOU置信度，大于该值，则认为是TP
        :param prob_threshold: 最小置信度
        """
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold

    def evaluate(self, gt_results: dict, dt_results: dict, class_name: list, plot=False):
        """
        :param gt_results: ground-truth results,format is
                gt_results={"class_name1":
                                    {"file-name1":[[score,x1,y1,x2,y2],[score,x1,y1,x2,y2],[]...]}
                                    {"file-name2":[[score,x1,y1,x2,y2],[score,x1,y1,x2,y2],[]...]}，
                            "class_name2":
                                    {"file-name1":[[score,x1,y1,x2,y2],[score,x1,y1,x2,y2],[]...]}
                                    {"file-name2":[[score,x1,y1,x2,y2],[score,x1,y1,x2,y2],[]...]}
                            }
        :param dt_results: detection-results，format is the same with gt_results
        :param class_name: 需要测评的类别
        :param plot: 是否显示绘图结果
        :return:
        """
        ap_logs = "Average Precision Per-class:\n"
        results = {}
        for name in class_name:
            if name not in gt_results or name not in dt_results:
                continue
            r = measurements.compute_average_precision_per_class(gt_results[name],
                                                                 dt_results[name],
                                                                 self.iou_threshold,
                                                                 use_2007_metric=False,
                                                                 name=name
                                                                 )
            results[name] = r
        aps = {k: v["ap"] for k, v in results.items()}
        map = sum(aps.values()) / len(aps)
        ap_logs += "{}\n".format(aps)
        ap_logs += "Average Precision Across All Classes:\nmAP:{:3.5f}".format(map)
        print(ap_logs)
        print("==" * 20)
        if plot and map > 0:
            measurements.plot_map_recall_precision(results)
        return aps, map

    def evaluate_for_json(self, dt_info="dt_info.json", gt_info="gt_info.json"):
        class_name = ['BACKGROUND', 'hand']
        dt_results = json_utils.read_json_data(dt_info)
        gt_results = json_utils.read_json_data(gt_info)
        self.evaluate(gt_results, dt_results, class_name, plot=False)


if __name__ == "__main__":
    val = MapMetric()
    val.evaluate_for_json()
