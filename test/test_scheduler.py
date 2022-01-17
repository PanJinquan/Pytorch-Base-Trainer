# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-08-13 10:08:07
"""
import torch.optim as optim
from torchvision import models, transforms
from basetrainer.scheduler import build_scheduler
from basetrainer.scheduler.MultiStepLR import MultiStepLR
from basetrainer.scheduler.CosineAnnealingLR import CosineAnnealingLR
from basetrainer.utils import plot_utils

if __name__ == "__main__":
    num_epochs = 100
    num_warn_up = 3
    lr_init = 0.1
    num_steps = 10
    milestones = [20, 30, 40]
    model = models.resnet18(pretrained=False)
    optimizer = optim.SGD(model.parameters(), lr=lr_init)
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
