# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-28 09:09:32
"""
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from ..utils import torch_data, torch_tools
from ..engine import engine
from ..engine import comm


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
        self.num_steps = 0  # 每个epoch的迭代次数
        self.num_samples = 0  # 训练的样本数目num_samples=num_steps*batch_size
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
        self.model = self.build_model(cfg, **kwargs)
        self.optimizer = self.build_optimizer(cfg, **kwargs)
        self.criterion = self.build_criterion(cfg, **kwargs)
        self.callbacks = self.build_callbacks(cfg, **kwargs)

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
        return torch_data.build_dataloader(dataset,
                                           batch_size,
                                           num_workers,
                                           shuffle=shuffle,
                                           phase=phase,
                                           distributed=distributed,
                                           **kwargs)

    @classmethod
    def build_model_parallel(cls, model: nn.Module, device_ids=None, distributed=True, **kwargs) -> nn.Module:
        return torch_data.build_model_parallel(model, device_ids, distributed=distributed, **kwargs)
