# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-30 17:04:51
"""

import torch.nn as nn
import torch.utils.data as torch_utils
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from ..engine import comm
from .torch_tools import get_torch_version


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
        if get_torch_version() >= 1.7:
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
