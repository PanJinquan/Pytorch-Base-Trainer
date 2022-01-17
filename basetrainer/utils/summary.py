# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-02 14:59:42
"""
import os
import sys
import tensorboardX as tensorboard
# from torch.utils import tensorboard


class SummaryWriter():
    def __init__(self, log_dir, *args, **kwargs):
        self.tensorboard = None
        if log_dir:
            # 修复tensorboard版本BUG
            self.tensorboard = tensorboard.SummaryWriter(log_dir, *args, **kwargs)

    def add_scalar(self, *args, **kwargs):
        if self.tensorboard:
            self.tensorboard.add_scalar(*args, **kwargs)

    def add_scalars(self, *args, **kwargs):
        if self.tensorboard:
            self.tensorboard.add_scalars(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.tensorboard:
            self.tensorboard.add_image(*args, **kwargs)


if __name__ == '__main__':
    main_process = True
    log_root = "./"
    writer1 = SummaryWriter(log_root if main_process else None)
    # main_process=False
    writer2 = SummaryWriter(log_root if main_process else None)
    writer3 = SummaryWriter(log_root if main_process else None)
    epochs = 200
    for epoch in range(epochs):
        if writer3:
            print(writer3)
        writer1.add_scalar("lr_epoch", epoch, epoch)
        writer2.add_scalar("lr_epoch", epoch, epoch)
        writer3.add_scalar("lr_epoch", epoch, epoch)
