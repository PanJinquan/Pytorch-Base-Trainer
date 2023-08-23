# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: FaceRecognitionTest
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-03-19 09:19:18
# --------------------------------------------------------
"""

import numpy as np
from . import iou


class MatchingBBoxes(object):
    """bboxes matching"""

    @staticmethod
    def get_matching_map(pre_bboxes, true_bboxes, iou_threshold=0., axis=1):
        """
        输入IOU矩阵,根据IOU进行位置匹配,返回满足阈值的匹配的映射关系表
        :param pre_bboxes,
        :param true_bboxes
        :param iou_threshold: 0:匹配所有true_bboxes,
                           1e-5:匹配iou>= 1e-5所有true_bboxes
        :param axis:
        :return: max_iou  : 匹配分数,<class 'tuple'>: ((len(true_data),)
                 pt_map: 匹配的映射关系表,<class 'tuple'>: (len(true_data), 2)
                            index_max[:, 0]: pred_index
                            index_max[:, 1]: true_index = [0,1,2,...,len(true_data)]是下标从0开始的递增序列
        """
        # 获得IOU矩阵iou_mat:其中纵轴表示true_data,横轴表示pre_data
        iou_mat = iou.get_iou_mat(true_bboxes, pre_bboxes)  # <class 'tuple'>: (len(true_bboxes), len(pre_bboxes))
        pred_index = np.argmax(iou_mat, axis=axis)  # max_index,<class 'tuple'>: (len(true_bboxes),)
        true_index = np.arange(0, len(pred_index))
        pt_map = np.asarray((pred_index, true_index)).T
        max_iou = np.max(iou_mat, axis=axis)
        index = max_iou >= iou_threshold
        max_iou = max_iou[index]
        pt_map = pt_map[index]
        return max_iou, pt_map

    @staticmethod
    def get_matching_index(pred_bboxes, true_bboxes, iou_threshold=0):
        """
        BUG:当iou_threshold为0时,pred_bboxes, true_bboxes维度不变,
            但当iou_threshold>0,pred_bboxes, true_bboxes输出维度可能变小,
        :param pred_bboxes:
        :param true_bboxes:
        :param iou_threshold:
        :return: 返回匹配关系:
                    pred_index: 匹配后对应pred_bboxes的index
                    true_index: 匹配后对应true_bboxes的index,[0,1,2,...,len(true_data)]是下标从0开始的递增序列        
        """
        if not isinstance(pred_bboxes, np.ndarray):
            pred_bboxes = np.asarray(pred_bboxes)
        if not isinstance(true_bboxes, np.ndarray):
            true_bboxes = np.asarray(true_bboxes)
        max_iou, pt_map = MatchingBBoxes.get_matching_map(pred_bboxes, true_bboxes, iou_threshold)
        pred_index = pt_map[:, 0]
        true_index = pt_map[:, 1]
        return pred_index, true_index

    @staticmethod
    def bboxes_matching(pred_bboxes, true_bboxes, iou_threshold=0):
        """
        BUG:当iou_threshold为0时,pred_bboxes, true_bboxes维度不变,
            但当iou_threshold>0,pred_bboxes, true_bboxes输出维度可能变小,
        :param pred_bboxes:
        :param true_bboxes:
        :param iou_threshold:
        :return:
        """
        if not isinstance(pred_bboxes, np.ndarray):
            pred_bboxes = np.asarray(pred_bboxes)
        if not isinstance(true_bboxes, np.ndarray):
            true_bboxes = np.asarray(true_bboxes)
        max_iou, pt_map = MatchingBBoxes.get_matching_map(pred_bboxes, true_bboxes, iou_threshold)
        pred_index = pt_map[:, 0]
        true_index = pt_map[:, 1]
        pred_bboxes = pred_bboxes[pred_index]
        true_bboxes = true_bboxes[true_index]
        return pred_bboxes, true_bboxes

    @staticmethod
    def bboxes_id_matching(pred_bboxes, true_bboxes, pred_id, iou_threshold=0.):
        """
        BUG:当iou_threshold为0时,face_bboxes, body_bboxes维度不变,
            但当iou_threshold>0,face_bboxes, body_bboxes输出维度可能变小,
        :param pred_bboxes:
        :param pred_id
        :param true_bboxes:
        :param iou_threshold:
        :return:
        """
        if not isinstance(pred_bboxes, np.ndarray):
            pred_bboxes = np.asarray(pred_bboxes)
        if not isinstance(true_bboxes, np.ndarray):
            true_bboxes = np.asarray(true_bboxes)
        if not isinstance(pred_id, np.ndarray):
            pred_id = np.asarray(pred_id)
        max_iou, pt_map = MatchingBBoxes.get_matching_map(pred_bboxes, true_bboxes, iou_threshold)
        pred_index = pt_map[:, 0]
        true_index = pt_map[:, 1]
        pred_bboxes = pred_bboxes[pred_index]
        true_bboxes = true_bboxes[true_index]
        pred_id = pred_id[pred_index]
        return pred_bboxes, true_bboxes, pred_id

    @staticmethod
    def face_body_matching(face_bboxes, body_bboxes, body_keys, face_keys, iou_threshold=1e-5, **kwargs):
        """
        实现人脸/人体bboes匹配算法(PS:将人体框body_bboxes匹配到人脸框face_bboxes,因此不会修改face_bboxes顺序)
        :param face_bboxes:<np.ndarray, list>人脸框(被匹配的bboxes)
        :param body_bboxes:<np.ndarray, list>人体框(待匹配的bboxes)
        :param body_keys:<set>与body_bboxes绑定的信息的key
        :param face_keys:<set>与face_bboxes绑定的信息的key
        :param body_keys:<np.ndarray, list>人体框(待匹配的bboxes)
        :param iou_threshold:<float>,人脸/人体框匹配阈值,即IOU阈值最低分数
        :param kwargs:<dict, optional>跟人脸框对应的相关数据,长度必须与face_bboxes一样,且跟face_bboxes是一一对应的关系
                      比如一个人脸框,对应一个人脸识别ID,或者对应一个表情识别标签.格式如
                      kwargs= {"emotion": {"labels": [], "scores": []},
                               "others":  {"labels": []},
                                ...
                                ...
                                ...
                            }
        :return:返回匹配后的数据
        """
        if not isinstance(face_bboxes, np.ndarray):
            face_bboxes = np.asarray(face_bboxes)
        if not isinstance(body_bboxes, np.ndarray):
            body_bboxes = np.asarray(body_bboxes)
        # 匹配人体框到人脸框,获得index对应关系
        body_index, face_index = MatchingBBoxes.get_matching_index(body_bboxes,
                                                                   face_bboxes,
                                                                   iou_threshold)
        face_bboxes = face_bboxes[face_index]
        body_bboxes = body_bboxes[body_index]

        # 脸框依赖的数据,需要根据face_index进行调整,避免丢失对应关系
        # 人体框依赖的数据,需要根据body_index进行调整,避免丢失对应关系
        out_kwargs = {}
        for name, dict_val in kwargs.items():
            out_kwargs[name] = {}
            for k, v in dict_val.items():
                if not isinstance(v, np.ndarray):
                    v = np.asarray(v)
                # 按照数据依赖不同（人体框或人脸框）进行顺序调整
                if name in body_keys:
                    out_kwargs[name][k] = v[body_index]
                elif name in face_keys:
                    out_kwargs[name][k] = v[face_index]

        return face_bboxes, body_bboxes, out_kwargs
