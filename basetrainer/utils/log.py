# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @Date   : 2019-9-20 13:18:34
# --------------------------------------------------------
"""
import os
import datetime
import logging
import threading
import re
import time
from logging.handlers import TimedRotatingFileHandler


# from memory_profiler import profile


def singleton(cls):
    _instance_lock = threading.Lock()
    instances = {}

    def _singleton(*args, **kwargs):
        with _instance_lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]

    return _singleton


@singleton  # 使用singleton，会出现loger的level失效的问题
class CustomLogger(logging.Logger):
    def __init__(self, name="LOG", level="debug"):
        """
        Initialize the logger with a name and an optional level.
        Args:
            name:
            level: debug,info,warning,critical,fatal
        """
        super().__init__(name)
        # super(CustomLogger, self).__init__(name)
        self.setLevel(level=level)

    @staticmethod
    def levels(level):
        if level == 'debug':
            return logging.DEBUG
        if level == 'info':
            return logging.INFO
        if level == 'warning':
            return logging.WARN
        if level == 'critical':
            return logging.CRITICAL
        if level == 'fatal':
            return logging.FATAL
        return logging.DEBUG

    def setLevel(self, level):
        """
        Args:
            level: debug,info,warning,critical,fatal
        Returns:
        """
        level = self.levels(level)
        super().setLevel(level)

    @staticmethod
    def set_format(handler, format):
        # handler.suffix = "%Y%m%d"
        if format:
            logFormatter = logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(levelname)s: %(message)s",
                                             "%Y-%m-%d %H:%M:%S")
        else:
            logFormatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(logFormatter)

    def show_batch_tensor(self, title, batch_imgs, index=0):
        pass


class FileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False,
                 atTime=None):
        logging.handlers.BaseRotatingHandler.__init__(self, filename, 'a', encoding, delay)
        self.when = when.upper()
        self.backupCount = backupCount
        self.utc = utc
        self.atTime = atTime
        if self.when == 'S':
            self.interval = 1  # one second
            self.suffix = "%Y-%m-%d_%H-%M-%S"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(\.\w+)?$"
        elif self.when == 'M':
            self.interval = 60  # one minute
            self.suffix = "%Y-%m-%d_%H-%M"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(\.\w+)?$"
        elif self.when == 'H':
            self.interval = 60 * 60  # one hour
            self.suffix = "%Y-%m-%d_%H"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}(\.\w+)?$"
        elif self.when == 'D' or self.when == 'MIDNIGHT':
            self.interval = 60 * 60 * 24  # one day
            self.suffix = "%Y-%m-%d"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}(\.\w+)?$"
        elif self.when.startswith('W'):
            self.interval = 60 * 60 * 24 * 7  # one week
            if len(self.when) != 2:
                raise ValueError("You must specify a day for weekly rollover from 0 to 6 (0 is Monday): %s" % self.when)
            if self.when[1] < '0' or self.when[1] > '6':
                raise ValueError("Invalid day specified for weekly rollover: %s" % self.when)
            self.dayOfWeek = int(self.when[1])
            self.suffix = "%Y-%m-%d"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}(\.\w+)?$"
        elif self.when == 'Y':
            self.interval = 60 * 60 * 24 * 365  # one yes
            self.suffix = "%Y-%m-%d"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}(\.\w+)?$"
        else:
            raise ValueError("Invalid rollover interval specified: %s" % self.when)

        self.extMatch = re.compile(self.extMatch, re.ASCII)
        self.interval = self.interval * interval  # multiply by units requested
        # The following line added because the filename passed in could be a
        # path object (see Issue #27493), but self.baseFilename will be a string
        filename = self.baseFilename
        if os.path.exists(filename):
            t = os.stat(filename)[logging.handlers.ST_MTIME]
        else:
            t = int(time.time())
        self.rolloverAt = self.computeRollover(t)


def set_logger(name="LOG", level="debug", logfile=None, format=False, is_main_process=True):
    """
    logger = set_logging(name="LOG", level="debug", logfile="log.txt", format=False)
    url:https://cuiqingcai.com/6080.html
    level级别：debug>info>warning>error>critical
    :param level: 设置log输出级别
    :param logfile: log保存路径，如果为None，则在控制台打印log
    :param is_main_process: 是否是主进程
    :return:
    """
    if not is_main_process:
        level = "fatal"
        logfile = None
    # logger = logging.getLogger(name)
    logger = CustomLogger(name, level=level)
    if logfile and os.path.exists(logfile):
        os.remove(logfile)
    # define a FileHandler write messages to file
    if logfile:
        # filehandler = logging.handlers.RotatingFileHandler(filename="./log.txt")
        # filehandler = TimedRotatingFileHandler(logfile, when="midnight", interval=1)
        filehandler = FileHandler(logfile, when="Y", interval=1)
        logger.set_format(filehandler, format)
        logger.addHandler(filehandler)

    # define a StreamHandler print messages to console
    console = logging.StreamHandler()
    logger.set_format(console, format)
    logger.addHandler(console)
    return logger


def print_args(args):
    logger = get_logger()
    logger.info("---" * 10)
    args = args.__dict__
    for k, v in args.items():
        # print("{}: {}".format(k, v))
        logger.info("{}: {}".format(k, v))
    logger.info("---" * 10)


def get_logger(name="LOG", level="debug"):
    logger = CustomLogger(name)
    # if logger.isEnabledFor(CustomLogger.levels(level)):
    #     logger = CustomLogger(name, level=level)
    return logger


def RUN_TIME(deta_time):
    '''
    计算时间差，返回毫秒,deta_time.seconds获得秒数=1000ms，deta_time.microseconds获得微妙数=1/1000ms
    :param deta_time: ms
    :return:
    '''
    time_ = deta_time.seconds * 1000 + deta_time.microseconds / 1000.0
    return time_


def TIME():
    '''
    获得当前时间
    :return:
    '''
    return datetime.datetime.now()


def run_time_decorator(title=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # torch.cuda.synchronize()
            T0 = TIME()
            result = func(*args, **kwargs)
            # torch.cuda.synchronize()
            T1 = TIME()
            print("{}-- function : {}-- rum time : {}ms ".format(title, func.__name__, RUN_TIME(T1 - T0)))
            # logger.debug("{}-- function : {}-- rum time : {}s ".format(title, func.__name__, RUN_TIME(T1 - T0)/1000.0))
            return result

        return wrapper

    return decorator


# @profile(precision=4)
def memory_test():
    """
    1.先导入：
    > from memory_profiler import profile
    2.函数前加装饰器：
    > @profile(precision=4,stream=open('memory_profiler.log','w+'))
　　　参数含义：precision:精确到小数点后几位
　　　stream:此模块分析结果保存到 'memory_profiler.log' 日志文件。如果没有此参数，分析结果会在控制台输出
    :return:
    """
    c = 0
    for item in range(10):
        c += 1
        # logger.error("c:{}".format(c))
    # print(c)


if __name__ == '__main__':
    # logger = set_logger(name="LOG", level="warning", logfile="log.txt", format=False)
    # T0 = TIME()
    # do something
    # T1 = TIME()
    # print("rum time:{}ms".format(RUN_TIME(T1 - T0)))
    # t_logger = set_logging(name=__name__, level="info", logfile=None)
    # t_logger.debug('debug')
    # t_logger.info('info')
    # t_logger.warning('Warning exists')
    # t_logger.error('Finish')
    # memory_test()
    # logger1 = set_logger(name="LOG", level="debug", logfile="log.txt", format=False)
    logger = set_logger(logfile=None, level="debug")
    logger1 = get_logger()
    logger1.info("---" * 20)
    logger1.debug("work_space:{}".format("work_dir"))
    logger1.info("work_space:{}".format("work_dir"))
    logger1.error("work_space:{}".format("work_dir"))
    logger1.fatal("work_space:{}".format("work_dir"))
    # logger1.show_batch_tensor()
