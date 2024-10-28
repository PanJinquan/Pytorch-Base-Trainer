# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-28 14:57:10
"""
import sys
import os
import torch
from ..callbacks.callbacks import Callback
from ..utils import summary
from ..utils import log
from pybaseutils import file_utils


class ModelCheckpoint(Callback):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim,
                 moder_dir: str,
                 epochs: int,
                 start_save: int = -1,
                 best_save: int = -1,
                 indicator: str = "",
                 inverse: bool = False,
                 logger=None,
                 **kwargs
                 ):
        """
        保存模型回调函数
        :param model: 模型
        :param optimizer:优化器
        :param moder_dir:保存训练模型的目录
        :param epochs: 训练的epochs数
        :param start_save: epoch >= start_save开始保存，如果为-1，则保存最后model_nums个epoch模型
        :param best_save: 最优模型开始保存
        :param indicator:需要关注的指标，以便保存最优模型，需要根据Metrics定义的指标对应，
                         如分类模型中indicator="acc"；如果关注losss,则indicator="loss"
                         如果不需要关注，则设置为空
                         PS: 优先关注"test"的指标
        :param inverse: 关注的指标(indicator)值越大越好，则inverse=True；值越小越好，则inverse=False
        :param logger: Log实例对象
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.moder_dir = moder_dir
        self.epochs = epochs
        self.start_save = start_save if start_save else -1
        self.best_save = best_save if best_save else -1
        file_utils.create_dir(self.moder_dir)
        self.logger = log.get_logger() if logger is None else logger
        self.main_process = True
        self.indicator = indicator
        self.indicator_val = 0
        self.inverse = inverse
        self.model_nums = kwargs.get("model_nums", 10)
        if self.inverse:
            self.indicator_val = sys.maxsize  # 评价指标

    def get_indicators_values(self, indicator, logs):
        r = None
        for k, v in logs.items():
            if k == indicator:
                return v
            elif isinstance(v, dict):
                r = self.get_indicators_values(indicator, v)
        return r

    def _sum(self, value):
        if isinstance(value, dict):
            # 去除total的数据，避免重复计算求和
            l = {k: v for k, v in value.items() if "total_" not in k}
            l = sum(l.values())
        elif isinstance(value, tuple) or isinstance(value, list):
            l = sum(value)
        else:
            l = value
        return l

    def on_epoch_end(self, epoch, logs: dict = {}):
        # fix a bug:
        if not self.is_main_process: return
        if "test" in logs:
            value = self.get_indicators_values(self.indicator, logs["test"])
        else:
            value = self.get_indicators_values(self.indicator, logs)
        value = self._sum(value)
        self.save_model(self.moder_dir, value, epoch, start_save=self.start_save)
        if value:
            self.save_best_model(self.moder_dir, value, epoch, self.inverse)

    def save_model(self, model_root, value, epoch, start_save=0):
        """保存模型"""
        model = self.model
        optimizer = self.optimizer
        if value:
            name = "model_{:0=3d}_{:.4f}.pth".format(epoch, value)
        else:
            name = "model_{:0=3d}.pth".format(epoch)
        # 保存当前优化器和模型
        torch.save({"epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   os.path.join(model_root, "model_optimizer.pth"))
        # 保存最新的模型
        model_file = os.path.join(model_root, "latest_{}".format(name))
        file_utils.remove_prefix_files(model_root, "latest_*")
        torch.save(model.module.state_dict(), model_file)
        # 保存最后epoch模型
        start_save = start_save if start_save >= 0 else self.epochs - self.model_nums
        if epoch >= start_save:
            model_file = os.path.join(model_root, name)
            torch.save(model.module.state_dict(), model_file)
        self.logger.info("save model:{}".format(model_file))

    def save_best_model(self, model_root, value, epoch, inverse=False):
        """保存关注的指标(indicator)最优的模型"""
        if epoch < self.best_save: return
        model = self.model
        optimizer = self.optimizer
        if value > self.indicator_val and inverse:
            # indicator指标，值越小，性能越好
            return
        elif value < self.indicator_val and (not inverse):
            # indicator指标，值越大，性能越好
            return
        self.indicator_val = value
        model_file = os.path.join(model_root, "best_model_{:0=3d}_{:.4f}.pth".format(epoch, value))
        file_utils.remove_prefix_files(model_root, "best_model_*")
        torch.save(model.module.state_dict(), model_file)
        self.logger.info("save best_model:{}".format(model_file))

    def resume_model(self, model, optimizer, resume, strict=True):
        """
        optimizer_pth = os.path.join(self.model_root, "optimizer_{}.pth".format(self.net_type))
        resume or finetune model
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, optimizer_pth)
        :return:
        """
        start_epoch = 0
        if os.path.isdir(resume):
            optimizer_pth = os.path.join(self.moder_dir, "model_optimizer.pth")
            model, optimizer, start_epoch = self._resume_model(optimizer_pth, model, optimizer)
            self.logger.info("resume_model:{},start_epoch:{}".format(optimizer_pth, start_epoch))
        elif os.path.isfile(resume):
            state_dict = torch.load(resume, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=strict)
            self.logger.info("pretrain_model:{},start_epoch:{},strict:{}".format(resume, start_epoch, strict))
        else:
            self.logger.info("no resume_model:{},start_epoch:{}".format(resume, start_epoch))
        return model, optimizer, start_epoch

    def _resume_model(self, optimizer_pth, model, optimizer):
        """
        optimizer_pth = os.path.join(self.model_root, "optimizer_{}.pth".format(self.net_type))
        resume or finetune model
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, optimizer_pth)
        :return:
        """
        state_dict = torch.load(optimizer_pth, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        # load optimizer parameter
        optimizer.load_state_dict(state_dict["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v
        start_epoch = state_dict["epoch"]
        return model, optimizer, start_epoch
