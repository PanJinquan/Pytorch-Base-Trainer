# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-01-25 18:39:09
"""
import numpy as np


class MultiStepValue(object):
    def __init__(self,
                 epochs,
                 steps_per_epoch,
                 milestones,
                 val_list=[],
                 val_init=0.01,
                 val_end=0,
                 num_warn_up=0,
                 decay_rates=None):
        """
        :param epochs:
        :param steps_per_epoch:
        :param lr_min : lr_min
        :param val_init: lr_max is init lr.
        :param num_warn_up:
        """
        super(MultiStepValue, self).__init__()
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.max_step = epochs * self.steps_per_epoch
        self.val_init = val_init
        self.val_end = val_end
        self.milestones = milestones
        self.warmup = num_warn_up * self.steps_per_epoch
        if not val_list:
            if decay_rates:
                val_list = [val_init * decay for decay in decay_rates]
            else:
                val_list = [val_init * 0.1 ** decay for decay in range(0, len(self.milestones) + 1)]
            if val_end is not None:
                val_list[-1] = val_end
        self.val_list = val_list
        self.val = self.val_init
        print("val_list:{}".format(self.val_list))
        print("milestones:{}".format(self.milestones))
        print("val_init:{}".format(self.val_init))
        assert len(self.milestones) == len(self.val_list) - 1

    def get_val(self):
        return self.val

    def __get_val(self, epoch, lr_stages, lr_list):
        """
        :param epoch:
        :param lr_stages:
        :param lr_list:
        :return:
        """
        lr = None
        max_stages = 0
        if not lr_stages:
            lr = lr_list[0]
        else:
            max_stages = max(lr_stages)
        for index in range(len(lr_stages)):
            if epoch < lr_stages[index]:
                lr = lr_list[index]
                break
            if epoch >= max_stages:
                lr = lr_list[index + 1]
        return lr

    def __set_stages_val(self, epoch, lr_stages, lr_list):
        '''
        :param epoch:
        :param lr_stages: [    35, 65, 95, 150]
        :param lr_list:   [0.1, 0.01, 0.001, 0.0001, 0.00001]
        :return:
        '''
        lr = self.__get_val(epoch, lr_stages, lr_list)
        if lr is not None:
            self.val = lr

    def set_val(self, epoch, total_step):
        """
        Usage:
        for epoch in range(epochs):
            for i in range(steps_per_epoch):
                scheduler.on_step(steps_per_epoch * epoch + i)
                ...
        :param total_step: total step: steps_per_epoch * epoch + step
        :return:
        """
        if self.warmup and total_step <= self.warmup:
            self.val = self.val_init / self.warmup * total_step
        else:
            self.__set_stages_val(epoch, self.milestones, self.val_list)

    def step(self, epoch, step=0):
        total_step = self.steps_per_epoch * epoch + step
        self.set_val(epoch, total_step)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Plot lr schedule
    epochs = 200
    steps_per_epoch = 100
    num_warn_up = 0
    milestones = list(range(15, epochs - 15, 15))
    val_list = np.arange(1, 0, -1 / len(milestones)).tolist() + [0]
    val_init = 1.0

    kd_decay_scheduler = MultiStepValue(epochs, steps_per_epoch,
                                        val_list=val_list,
                                        val_init=val_init,
                                        milestones=milestones,
                                        num_warn_up=num_warn_up)
    lr_steps = []
    lr_epochs = []
    for epoch in range(epochs):
        # scheduler.epoch(epoch)
        # for i in range(steps_per_epoch):
        kd_decay_scheduler.step(epoch)
        lr = kd_decay_scheduler.get_val()
        lr_steps.append(lr)
        lr_epochs.append(lr)
        print(epoch, lr)
    plt.figure()
    plt.plot(lr_steps, label="lr_steps")
    plt.grid(True)  # 显示网格;
    plt.xlabel("steps")
    plt.ylabel("LR")

    plt.figure()
    plt.plot(lr_epochs, label="lr_epochs")
    plt.grid(True)  # 显示网格;
    plt.xlabel("epochd")
    plt.ylabel("LR")

    plt.show()
