# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-08-13 10:08:07
"""
import torch
import torch.optim as optim
from torchvision import models, transforms
from basetrainer.scheduler import build_scheduler
from basetrainer.scheduler.MultiStepLR import MultiStepLR
from basetrainer.scheduler.CosineAnnealingLR import CosineAnnealingLR
from basetrainer.utils import plot_utils


def test_callback_for_build_scheduler(num_epochs, num_steps, optimizer, lr_init, milestones, num_warn_up=10):
    # scheduler = "ExpLR"  # cosine,multi-step
    # scheduler = "LambdaLR"  # cosine,multi-step
    scheduler = "cosine"  # cosine,multi-step
    lr_scheduler = build_scheduler.get_scheduler(scheduler,
                                                 optimizer=optimizer,
                                                 lr_init=lr_init,
                                                 num_epochs=num_epochs,
                                                 num_steps=num_steps,
                                                 milestones=milestones,
                                                 num_warn_up=num_warn_up)

    lr_list = []
    for epoch in range(num_epochs):
        lr_scheduler.on_epoch_begin(epoch)
        for step in range(num_steps):
            lr_scheduler.step(epoch, step)
        lr = optimizer.param_groups[0]["lr"]
        lr_list.append(lr)
        print("epoch:{},lr:{}".format(epoch, lr))
    plot_utils.plot_multi_line(x_data_list=[range(num_epochs)], y_data_list=[lr_list])


def test_torch_scheduler(num_epochs, num_steps, optimizer, lr_init, milestones, num_warn_up):
    # 余弦退火学习率,T_0是周期，T_mult就是之后每个周期T_0 = T_0 * T_mult，eta_min最低学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2,eta_min=1e-5)
    # num_cycles = 3
    # t_max = num_epochs * 1.0 / (2 * num_cycles - 1)  # 一次学习率周期的迭代次数，即 T_max 个 epoch 之后重新设置学习率。
    # eta_min = 0.000001  # 最小学习率，即在一个周期中，学习率最小会下降到 eta_min，默认值为 0
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=eta_min, last_epoch=-1)

    lr_list = []
    for epoch in range(num_epochs):
        # lr_scheduler.on_epoch_begin(epoch)
        for step in range(num_steps):
            # lr_scheduler.step(epoch, step)
            pass
        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        lr_list.append(lr)
        print("epoch:{},lr:{}".format(epoch, lr))
    plot_utils.plot_multi_line(x_data_list=[range(num_epochs)], y_data_list=[lr_list])


if __name__ == "__main__":
    num_epochs = 60
    num_warn_up = 10
    lr_init = 0.001
    num_steps = 1000
    milestones = []
    # milestones = [50, 100, 120]
    model = models.resnet18(pretrained=False)
    optimizer = optim.SGD(model.parameters(), lr=lr_init)
    test_callback_for_build_scheduler(num_epochs, num_steps, optimizer, lr_init, milestones, num_warn_up)
    # test_torch_scheduler(num_epochs, num_steps, optimizer, lr_init, milestones, num_warn_up)
