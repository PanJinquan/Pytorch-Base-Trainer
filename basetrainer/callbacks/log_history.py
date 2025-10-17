# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-28 14:57:10
"""
import os
import csv
from ..callbacks.callbacks import Callback
from ..utils import summary
from ..utils import log
from pybaseutils import file_utils, pandas_utils


class LogHistory(Callback):
    def __init__(self, log_dir, log_freq=1, logger=None, indicators: list = [], is_main_process=True):
        """
        Tensorboard,Log等summary记录信息
        :param log_dir:Log输出日志保存目录
        :param log_freq:Log打印频率
        :param logger:Log实例对象，如果logger=None，则会初始化新的Log实例对象
        :param indicators: 需要Tensorboard记录的指标,由recorder回调函数指定,可指定多个,如["Acc","loss"]
        :param is_main_process: 是否是主进程，仅当在主进程中才会打印Log信息
        """
        super().__init__()
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.is_main_process = is_main_process
        if isinstance(indicators, str): indicators = [indicators]
        self.indicators = list(set(indicators))
        # Log实例对象
        file_utils.create_dir(self.log_dir)
        self.logfile = os.path.join(self.log_dir, "history.log")
        self.csvfile = os.path.join(self.log_dir, "results.csv")
        self.logger = log.set_logger(logfile=self.logfile) if logger is None else logger
        self.csv_writer, self.csv_titles = None, []
        if self.is_main_process:
            self.csv_titles = (['epoch'] +
                               [f"train-{n}" for n in self.indicators] +
                               [f"test-{n}" for n in self.indicators])
            self.csv_writer = pandas_utils.CSVWriter(self.csvfile, title=self.csv_titles)
        # 初始化Tensorboard
        self.log_writer = summary.SummaryWriter(log_dir=self.log_dir if self.is_main_process else None)

    def on_batch_end(self, batch, logs: dict = {}):
        if batch % self.log_freq == 0 or batch == 0:
            info = dict(logs)
            info.pop("test") if "test" in logs else info
            self.logger.info(info)

    def get_indicators_values(self, indicator, logs):
        for k, v in logs.items():
            if k == indicator:
                return v
            elif isinstance(v, dict):
                return self.get_indicators_values(indicator, v)
        return None

    @staticmethod
    def summary(phase, log_writer, indicators, epoch, logs: dict = {}):
        output = []
        for indicator in indicators:
            if not indicator in logs[phase]:
                output.append(None)
                continue
            if isinstance(logs[phase][indicator], dict):
                log_writer.add_scalars(main_tag="{}_{}_epoch".format(phase, indicator),
                                       tag_scalar_dict=logs[phase][indicator],
                                       global_step=epoch)
            else:
                log_writer.add_scalar("{}_{}_epoch".format(phase, indicator), logs[phase][indicator], epoch)
            output.append(logs[phase][indicator])
        return output

    def on_epoch_end(self, epoch, logs: dict = {}):
        train_logs = [None] * len(self.indicators)
        test_logs = [None] * len(self.indicators)
        if "train" in logs:
            train_logs = self.summary(phase="train", log_writer=self.log_writer, indicators=self.indicators,
                                      epoch=epoch, logs=logs)
            self.logger.info("train epoch:{:0=3},{}".format(epoch, logs["train"]))
            self.log_writer.add_scalar("lr_epoch", logs["lr"], epoch)
        if "test" in logs:
            test_logs = self.summary(phase="test", log_writer=self.log_writer, indicators=self.indicators,
                                     epoch=epoch, logs=logs)
            self.logger.info("test epoch:{:0=3},{}".format(epoch, logs["test"]))
        csv_data = [epoch] + train_logs + test_logs
        if self.csv_writer:
            self.csv_writer.add(csv_data)
