# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-28 09:09:32
"""
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from ..utils import torch_data, torch_tools
from ..engine import engine
from ..engine import comm

print("torch version:{}".format(torch.__version__))
torch.multiprocessing.set_sharing_strategy('file_system')


class EngineTrainer(engine.Engine):
    def __init__(self, cfg):
        super(EngineTrainer, self).__init__(cfg)
        torch_tools.set_env_random_seed()
        self.model = None  # 模型
        self.optimizer = None  # 优化器
        self.criterion = None  # 损失函数
        self.train_loader = None  # 训练数据loader
        self.test_loader = None  # 测试数据loader
        self.num_epochs = cfg.num_epochs  # 数据循环迭代次数
        self.train_num = 0  # 训练集的样本数目num_samples=num_steps*batch_size
        self.test_num = 0  # 测试集的样本数目num_samples=num_steps*batch_size
        self.num_samples = 0  # self.train_nums
        self.num_steps = 0  # 每个epoch的迭代次数
        self.criterion = None  # 损失函数
        self.progress = cfg.progress  # 是否显示进度条
        self.is_main_process = comm.is_main_process()  # 分布式训练中，是否是主进程
        self.world_size = comm.get_world_size()  # 分布式训练中，开启的工作进程数，一般等于GPU卡数
        self.local_rank = comm.get_local_rank()  # 分布式训练中，工作进程ID
        self.device = comm.get_device(cfg.gpu_id)  # 分布式训练中，当前工作进程使用的device

    def build(self, cfg, **kwargs):
        self.train_loader = self.build_train_loader(cfg, **kwargs)
        self.test_loader = self.build_test_loader(cfg, **kwargs)
        if self.train_loader:
            self.num_steps = len(self.train_loader)
            self.num_samples = len(self.train_loader.sampler)
            self.train_num = len(self.train_loader.sampler)
        if self.test_loader:
            self.test_num = len(self.test_loader.sampler)
        self.model = self.build_model(cfg, **kwargs)
        self.criterion = self.build_criterion(cfg, **kwargs)
        self.optimizer = self.build_optimizer(cfg, **kwargs)
        self.callbacks = self.build_callbacks(cfg, **kwargs)
        self.callbacks = [c for c in self.callbacks if c]

    def build_train_loader(self, cfg, **kwargs):
        """定义训练数据"""
        raise NotImplementedError("build_train_loader not implemented!")

    def build_test_loader(self, cfg, **kwargs):
        """定义测试数据"""
        raise NotImplementedError("build_test_loader not implemented!")

    def build_model(self, cfg, **kwargs):
        """定于训练模型"""
        raise NotImplementedError("build_model not implemented!")

    def build_optimizer(self, cfg, **kwargs):
        """定义优化器"""
        raise NotImplementedError("build_optimizer not implemented!")

    def build_criterion(self, cfg, **kwargs):
        """定义损失函数"""
        raise NotImplementedError("build_criterion not implemented!")

    def build_callbacks(self, cfg, **kwargs):
        """定义回调函数"""
        raise NotImplementedError("build_callbacks not implemented!")

    @classmethod
    def build_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int,
                         shuffle: bool = True, phase: str = "train", distributed=True,
                         **kwargs) -> DataLoader:
        """
        :param dataset: Dataset
        :param batch_size:
        :param num_workers:
        :param shuffle:
        :param persistent_workers: 该参数仅支持torch>=1.6
               False: 数据加载器运行完一个Epoch后会关闭worker进程,在分布式训练，会出现每个epoch初始化多进程的问题
               True: 会保持worker进程实例激活状态,容易出现“DataLoader worker (pid(s) 953) exited unexpectedly”的错误
        :param phase: "train", "test", "val"
        :param distributed: True: use DDP; False: use DP (是否使用分布式训练)
        :param kwargs:
        :return:
        """
        return torch_data.build_dataloader(dataset,
                                           batch_size,
                                           num_workers,
                                           shuffle=shuffle,
                                           phase=phase,
                                           distributed=distributed,
                                           **kwargs)

    @classmethod
    def build_model_parallel(cls, model: nn.Module, device_ids=None, distributed=True, **kwargs) -> nn.Module:
        """
        :param model:
        :param device_ids:
        :param distributed: True: use DDP; False: use DP (是否使用分布式训练)
        :param kwargs:
        :return:
        """
        return torch_data.build_model_parallel(model, device_ids, distributed=distributed, **kwargs)
