# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-08-12 20:31:19
"""
from .MultiStepLR import MultiStepLR
from .CosineAnnealingLR import CosineAnnealingLR
from .ExponentialLR import ExponentialLR
from .LambdaLR import LambdaLR


def get_scheduler(scheduler, optimizer, lr_init, num_epochs, num_steps, **kwargs):
    if scheduler.lower() == "multi-step".lower() or scheduler.lower() == "multi_step".lower():
        lr_scheduler = MultiStepLR(optimizer,
                                   lr_init=lr_init,
                                   epochs=num_epochs,
                                   num_steps=num_steps,
                                   milestones=kwargs["milestones"],
                                   num_warn_up=kwargs["num_warn_up"])
    elif scheduler.lower() == "cosine".lower():
        # 余弦退火学习率调整策略
        lr_scheduler = CosineAnnealingLR(optimizer,
                                         num_epochs,
                                         num_steps=num_steps,
                                         lr_init=lr_init,
                                         num_warn_up=kwargs["num_warn_up"],
                                         num_cycles=kwargs.get("num_cycles", 1),
                                         decay=kwargs.get("decay", 1.0),
                                         )
    elif scheduler.lower() == "ExpLR".lower() or scheduler.lower() == "exp".lower():
        # 指数衰减学习率
        lr_scheduler = ExponentialLR(optimizer,
                                     num_epochs,
                                     num_steps=num_steps,
                                     lr_init=lr_init,
                                     num_warn_up=kwargs["num_warn_up"],
                                     decay=kwargs.get("decay", 0.95),
                                     )
    elif scheduler.lower() == "LambdaLR".lower():
        # 指数衰减学习率
        lr_scheduler = LambdaLR(optimizer,
                                num_epochs,
                                num_steps=num_steps,
                                lr_init=lr_init,
                                num_warn_up=kwargs["num_warn_up"]
                                )
    else:
        raise Exception("Error: scheduler type: {}".format(scheduler))
    return lr_scheduler
