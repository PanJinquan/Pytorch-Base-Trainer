# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-28 09:09:32
"""

import torch
import torch.utils.data as torch_utils
from tqdm import tqdm
from ..engine import base


class Engine(base.Base):
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_epochs = 0
        self.start_epoch = 0
        self.callbacks = []
        self.model = None
        self.device = "cuda"
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None
        self.progress = True  # 是否显示进度条
        super(Engine, self).__init__()

    def to(self, device):
        self.device = device

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(self.model)

    def on_batch_begin(self, batch, logs: dict = {}):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs: dict = {}):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs: dict = {}):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs: dict = {}):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs: dict = {}):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: dict = {}):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs: dict = {}):
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs: dict = {}):
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_train_summary(self, inputs, outputs, losses, epoch, batch, logs: dict = {}):
        for callback in self.callbacks:
            callback.on_train_summary(inputs, outputs, losses, epoch, batch, logs)

    def on_test_summary(self, inputs, outputs, losses, epoch, batch, logs: dict = {}):
        for callback in self.callbacks:
            callback.on_test_summary(inputs, outputs, losses, epoch, batch, logs)

    def _device(self, inputs):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        elif isinstance(inputs, tuple):
            inputs = (self._device(v) for v in inputs)
        elif isinstance(inputs, list):
            inputs = [self._device(v) for v in inputs]
        elif isinstance(inputs, dict):
            inputs = {k: self._device(v) for k, v in inputs.items()}
        return inputs

    def _losses(self, losses):
        if isinstance(losses, dict):
            l = sum(losses.values())
        elif isinstance(losses, tuple) or isinstance(losses, list):
            l = sum(losses)
        else:
            l = losses
        return l

    def run(self, logs: dict = {}):
        self.model.to(self.device)
        self.on_train_begin(logs)
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.set_model(self.model)
            self.on_epoch_begin(epoch, logs)
            if self.train_loader:
                self.run_train_epoch(self.train_loader, logs)
            self.on_test_begin(logs)
            if self.test_loader:
                self.run_test_epoch(self.test_loader, logs)
            self.on_test_end(logs)
            self.on_epoch_end(epoch, logs)
        self.on_train_end(logs)

    def run_train_epoch(self, dataset: torch_utils.DataLoader, logs: dict = {}):
        num_steps = len(dataset)
        batch = 0
        self.model.train()  # set to training mode
        with tqdm(total=num_steps, desc="Train Epoch #{}".format(self.epoch), disable=not self.progress) as t:
            for inputs in dataset:
                self.on_batch_begin(batch, logs)
                inputs = self._device(inputs)
                outputs, losses = self.run_step(inputs)
                # compute gradient and do optimizer step
                self.optimizer.zero_grad()
                self._losses(losses).backward()
                self.optimizer.step()
                logs['lr'] = self.optimizer.param_groups[0]["lr"]
                self.on_train_summary(inputs, outputs, losses, self.epoch, batch, logs)
                self.on_batch_end(batch, logs)
                batch += 1
                t.update(1)

    def run_test_epoch(self, dataset: torch_utils.DataLoader, logs: dict = {}):
        num_steps = len(dataset)
        batch = 0
        self.model.eval()  # set to evaluates mode
        with tqdm(total=num_steps, desc="Test  Epoch #{}".format(self.epoch), disable=not self.progress) as t:
            for inputs in dataset:
                with torch.no_grad():
                    inputs = self._device(inputs)
                    outputs, losses = self.run_step(inputs)
                    self.on_test_summary(inputs, outputs, losses, self.epoch, batch, logs)
                batch += 1
                t.update(1)

    def run_step(self, inputs):
        data, targets = inputs
        outputs = self.model(data)
        losses = self.criterion(outputs, targets)
        return outputs, losses
