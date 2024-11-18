# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-30 17:04:51
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_utils
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from ..engine import comm
from .torch_tools import get_torch_version, torch_version_id


class Collation(object):
    """
    Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations.
    """

    def __init__(self, stacks={}):
        """
        :param stacks: 需要堆叠一个batch-size的数据，通过Key指定
                      如 stacks = {'image': True, 'target': True, 'points': False}
        """
        self.stacks = stacks

    def __call__(self, batch):
        if isinstance(batch[0], dict):
            return self.collate_for_dict(batch)
        elif isinstance(batch[0], tuple) or isinstance(batch[0], list):
            return self.collate_for_list_tuple(batch)
        else:
            return batch

    def collate_for_dict(self, batch):
        """Dataset返回的Dict格式的数据"""
        # 初始化outputs和lengths
        data = batch[0]
        outputs = {k: list() for k, v in data.items()}
        assert isinstance(data, dict), "batch's item must be Dict"
        if not self.stacks: self.stacks = {k: True for k, v in data.items()}
        # print("count:{},stacks:{}".format(self.count, self.stacks))
        # 将batch相同Key合并在同一个list中
        for data in batch:
            for k, v in data.items():
                if isinstance(v, list) or isinstance(v, tuple):
                    outputs[k].append(v)
                elif isinstance(v, np.ndarray):
                    outputs[k].append(torch.from_numpy(v))
                elif isinstance(v, int):
                    outputs[k].append(torch.tensor(v, dtype=torch.int64))
                elif isinstance(v, float):
                    outputs[k].append(torch.tensor(v, dtype=torch.float32))
                else:
                    outputs[k].append(v)
        # 仅当维度相同时，才进行stack
        for k, v in outputs.items():
            try:
                if self.stacks[k]: outputs[k] = torch.stack(outputs[k])
            except:
                self.stacks[k] = False
        return outputs

    def collate_for_list_tuple(self, batch):
        """Dataset返回的List或Tuple格式的数据"""
        # 初始化outputs和lengths
        data = batch[0]
        outputs = [[] for k, v in enumerate(data)]
        assert isinstance(data, tuple) or isinstance(data, list), "batch's item must be List or tuple"
        if not self.stacks: self.stacks = {k: True for k, v in enumerate(data)}
        # 将batch总，相同Key合并在同一个list中
        for data in batch:
            for k, v in enumerate(data):
                if isinstance(v, list) or isinstance(v, tuple):
                    v = np.asarray(v)
                if isinstance(v, np.ndarray):
                    outputs[k].append(torch.from_numpy(v))
                elif isinstance(v, int):
                    outputs[k].append(torch.tensor(v, dtype=torch.int64))
                elif isinstance(v, float):
                    outputs[k].append(torch.tensor(v, dtype=torch.float32))
                else:
                    outputs[k].append(v)
        # 仅当维度相同时，才进行stack
        for k, v in enumerate(outputs):
            try:
                if self.stacks[k]:  outputs[k] = torch.stack(outputs[k])
            except:
                self.stacks[k] = False
        return outputs


def build_dataloader(dataset: Dataset,
                     batch_size: int,
                     num_workers: int,
                     shuffle: bool = True,
                     persistent_workers: bool = False,
                     phase: str = "train",
                     distributed=True,
                     **kwargs) -> DataLoader:
    """
    :param dataset: Dataset
    :param batch_size: DP模式中，每张卡将分配(batch_size/GPU)数进行训练；而DDP模式中，每张卡都有batch_size的数据
    :param num_workers:
    :param shuffle: 是否打乱顺序
    :param persistent_workers: 该参数仅支持torch>=1.6
           False: 数据加载器运行完一个Epoch后会关闭worker进程,在分布式训练，会出现每个epoch初始化多进程的问题
           True: 会保持worker进程实例激活状态,容易出现“DataLoader worker (pid(s) 953) exited unexpectedly”的错误
    :param phase: "train", "test", "val"
    :param distributed: True: use DDP; False: use DP (是否使用分布式训练)
    :param kwargs:
    :return:
    """
    assert phase in ["train", "test", "val"]
    sampler = None
    if comm.get_world_size() > 1 and phase == "train" and distributed:
        # DistributedSampler为每个子进程分发数据，避免数据重复
        sampler = torch_utils.distributed.DistributedSampler(dataset,
                                                             num_replicas=comm.get_world_size(),
                                                             rank=comm.get_local_rank(),
                                                             shuffle=shuffle)
        shuffle = False  # sampler option is mutually exclusive with shuffle
    else:
        # Fix a Bug: RuntimeError: can't start new thread
        # Fix a Bug: DataLoader worker (pid(s) 953) exited unexpectedly
        persistent_workers = False
    try:
        # Fix a Bug: torch<=1.6 have no argument 'persistent_workers'
        if get_torch_version() >= torch_version_id("1.7"):
            kwargs["persistent_workers"] = persistent_workers
            # fix a bug: persistent_workers option needs num_workers > 0
            if persistent_workers and num_workers == 0:
                kwargs["persistent_workers"] = False
    except:
        print("torch<=1.6 have no argument persistent_workers")
    dataloader = torch_utils.DataLoader(dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        sampler=sampler,
                                        shuffle=shuffle,
                                        **kwargs)
    return dataloader


def build_model_parallel(model: nn.Module,
                         device_ids=None,
                         distributed=True,
                         **kwargs) -> nn.Module:
    """
    :param model:
    :param device_ids:
    :param distributed: True: use DDP; False: use DP (是否使用分布式训练)
    :param kwargs:
    :return:
    """
    print("device_ids:{},device:{}".format(device_ids, comm.get_device(device_ids)))
    model.to(comm.get_device(device_ids))
    # use DistributedDataParallel
    if comm.get_world_size() > 1 and distributed:
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[comm.get_device(device_ids)],
                                                    output_device=comm.get_device(device_ids),
                                                    **kwargs
                                                    )
    else:
        # use DataParallel
        model = nn.DataParallel(model, device_ids=device_ids, output_device=comm.get_device(device_ids), **kwargs)
    return model
