# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-07-28 09:09:32
"""
import os
import sys

sys.path.append(os.getcwd())
import argparse
import basetrainer
from torchvision import transforms
from torchvision.datasets import ImageFolder
from basetrainer.engine import trainer
from basetrainer.engine.launch import launch
from basetrainer.criterion.criterion import get_criterion
from basetrainer.metric import accuracy_recorder
from basetrainer.callbacks import log_history, model_checkpoint, losses_recorder
from basetrainer.scheduler import build_scheduler
from basetrainer.optimizer.build_optimizer import get_optimizer
from basetrainer.utils import log, file_utils, setup_config, torch_tools
from basetrainer.models import build_models

print(basetrainer.__version__)


class ClassificationTrainer(trainer.EngineTrainer):
    """ Training Pipeline """

    def __init__(self, cfg):
        super(ClassificationTrainer, self).__init__(cfg)
        torch_tools.set_env_random_seed()
        cfg.model_root = os.path.join(cfg.work_dir, "model")
        cfg.log_root = os.path.join(cfg.work_dir, "log")
        if self.is_main_process:
            file_utils.create_dir(cfg.work_dir)
            file_utils.create_dir(cfg.model_root)
            file_utils.create_dir(cfg.log_root)
            file_utils.copy_file_to_dir(cfg.config_file, cfg.work_dir)
            setup_config.save_config(cfg, os.path.join(cfg.work_dir, "setup_config.yaml"))
        self.logger = log.set_logger(level="debug",
                                     logfile=os.path.join(cfg.log_root, "train.log"),
                                     is_main_process=self.is_main_process)
        # build project
        self.build(cfg)
        self.logger.info("=" * 60)
        self.logger.info("work_dir          :{}".format(cfg.work_dir))
        self.logger.info("config_file       :{}".format(cfg.config_file))
        self.logger.info("gpu_id            :{}".format(cfg.gpu_id))
        self.logger.info("main device       :{}".format(self.device))
        self.logger.info("num_samples(train):{}".format(self.num_samples))
        self.logger.info("num_classes       :{}".format(cfg.num_classes))
        self.logger.info("mean_num          :{}".format(self.num_samples / cfg.num_classes))
        self.logger.info("=" * 60)

    def build_optimizer(self, cfg, **kwargs):
        """build_optimizer"""
        self.logger.info("build_optimizer")
        self.logger.info("optim_type:{},init_lr:{},weight_decay:{}".format(cfg.optim_type, cfg.lr, cfg.weight_decay))
        optimizer = get_optimizer(self.model,
                                  optim_type=cfg.optim_type,
                                  lr=cfg.lr,
                                  momentum=cfg.momentum,
                                  weight_decay=cfg.weight_decay)
        return optimizer

    def build_criterion(self, cfg, **kwargs):
        """build_criterion"""
        self.logger.info("build_criterion,loss_type:{},num_classes:{}".format(cfg.loss_type, cfg.num_classes))
        criterion = get_criterion(cfg.loss_type, cfg.num_classes, device=self.device)
        return criterion

    def build_train_loader(self, cfg, **kwargs):
        """build_train_loader"""
        self.logger.info("build_train_loader,input_size:{}".format(cfg.input_size))
        transform = transforms.Compose([
            transforms.Resize([int(128 * cfg.input_size[1] / 112), int(128 * cfg.input_size[0] / 112)]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop([cfg.input_size[1], cfg.input_size[0]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        dataset = ImageFolder(root=cfg.train_data, transform=transform)
        cfg.num_classes = len(dataset.classes)
        cfg.classes = dataset.classes
        loader = self.build_dataloader(dataset, cfg.batch_size, cfg.num_workers, phase="train",
                                       shuffle=True, pin_memory=False, drop_last=True, distributed=cfg.distributed)
        return loader

    def build_test_loader(self, cfg, **kwargs):
        """build_test_loader"""
        self.logger.info("build_test_loader,input_size:{}".format(cfg.input_size))
        transform = transforms.Compose([
            transforms.Resize([int(128 * cfg.input_size[1] / 112), int(128 * cfg.input_size[0] / 112)]),
            transforms.CenterCrop([cfg.input_size[1], cfg.input_size[0]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        dataset = ImageFolder(root=cfg.train_data, transform=transform)
        loader = self.build_dataloader(dataset, cfg.batch_size, cfg.num_workers, phase="test",
                                       shuffle=False, pin_memory=False, drop_last=False, distributed=False)
        return loader

    def build_model(self, cfg, **kwargs):
        """build_model"""
        self.logger.info("build_model,net_type:{}".format(cfg.net_type))
        model = build_models.get_models(net_type=cfg.net_type, input_size=cfg.input_size,
                                        num_classes=cfg.num_classes, pretrained=True)
        if cfg.finetune:
            self.logger.info("finetune:{}".format(cfg.finetune))
            state_dict = torch_tools.load_state_dict(cfg.finetune)
            model.load_state_dict(state_dict)
        if cfg.use_prune:
            from basetrainer.pruning import nni_pruning
            sparsity = 0.2
            self.logger.info("use_prune:{},sparsity:{}".format(cfg.use_prune, sparsity))
            model = nni_pruning.model_pruning(model,
                                              input_size=[1, 3, cfg.input_size[1], cfg.input_size[0]],
                                              sparsity=sparsity,
                                              reuse=False,
                                              output_prune=os.path.join(cfg.work_dir, "prune"))
        model = self.build_model_parallel(model, cfg.gpu_id, distributed=cfg.distributed)
        return model

    def build_callbacks(self, cfg, **kwargs):
        """定义回调函数"""
        self.logger.info("build_callbacks")
        # 准确率记录回调函数
        acc_record = accuracy_recorder.AccuracyRecorder(target_names=cfg.classes,
                                                        indicator="Accuracy")
        # loss记录回调函数
        loss_record = losses_recorder.LossesRecorder(indicator="loss")
        # Tensorboard Log等历史记录回调函数
        history = log_history.LogHistory(log_dir=cfg.log_root,
                                         log_freq=cfg.log_freq,
                                         logger=self.logger,
                                         indicators=["loss", "Accuracy"],
                                         is_main_process=self.is_main_process)
        # 模型保存回调函数
        checkpointer = model_checkpoint.ModelCheckpoint(model=self.model,
                                                        optimizer=self.optimizer,
                                                        moder_dir=cfg.model_root,
                                                        epochs=cfg.num_epochs,
                                                        start_save=-1,
                                                        indicator="Accuracy",
                                                        logger=self.logger)
        # 学习率调整策略回调函数
        lr_scheduler = build_scheduler.get_scheduler(cfg.scheduler,
                                                     optimizer=self.optimizer,
                                                     lr_init=cfg.lr,
                                                     num_epochs=cfg.num_epochs,
                                                     num_steps=self.num_steps,
                                                     milestones=cfg.milestones,
                                                     num_warn_up=cfg.num_warn_up)
        callbacks = [acc_record,
                     loss_record,
                     lr_scheduler,
                     history,
                     checkpointer]
        return callbacks

    def run(self, logs: dict = {}):
        self.logger.info("start train")
        super().run(logs)


def main(cfg):
    t = ClassificationTrainer(cfg)
    return t.run()


def get_parser():
    parser = argparse.ArgumentParser(description="Training Pipeline")
    parser.add_argument("-c", "--config_file", help="configs file", default="configs/config.yaml", type=str)
    # parser.add_argument("-c", "--config_file", help="configs file", default=None, type=str)
    parser.add_argument("--train_data", help="train data", default="./data/dataset/train", type=str)
    parser.add_argument("--test_data", help="test data", default="./data/dataset/val", type=str)
    parser.add_argument("--work_dir", help="work_dir", default="output", type=str)
    parser.add_argument("--input_size", help="input size", nargs="+", default=[224, 224], type=int)
    parser.add_argument("--batch_size", help="batch_size", default=32, type=int)
    parser.add_argument("--gpu_id", help="specify your GPU ids", nargs="+", default=[0], type=int)
    parser.add_argument("--num_workers", help="num_workers", default=0, type=int)
    parser.add_argument("--num_epochs", help="total epoch number", default=50, type=int)
    parser.add_argument("--scheduler", help=" learning scheduler: multi-step,cosine", default="multi-step", type=str)
    parser.add_argument("--milestones", help="epoch stages to decay learning rate", nargs="+",
                        default=[10, 20, 40], type=int)
    parser.add_argument("--num_warn_up", help="num_warn_up", default=3, type=int)
    parser.add_argument("--net_type", help="net_type", default="mobilenet_v2", type=str)
    parser.add_argument("--finetune", help="finetune model file", default=None, type=str)
    parser.add_argument("--loss_type", help="loss_type", default="CELoss", type=str)
    parser.add_argument("--optim_type", help="optim_type", default="SGD", type=str)
    parser.add_argument("--lr", help="learning rate", default=0.1, type=float)
    parser.add_argument("--weight_decay", help="weight_decay", default=0.0005, type=float)
    parser.add_argument("--momentum", help="momentum", default=0.9, type=float)
    parser.add_argument("--log_freq", help="log_freq", default=10, type=int)
    parser.add_argument('--use_prune', action='store_true', help='use prune', default=False)
    parser.add_argument('--progress', action='store_true', help='display progress bar', default=True)
    parser.add_argument('--distributed', action='store_true', help='use distributed training', default=False)
    parser.add_argument('--polyaxon', action='store_true', help='polyaxon', default=False)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    cfg = setup_config.parser_config(parser.parse_args(), cfg_updata=True)
    launch(main,
           num_gpus_per_machine=len(cfg.gpu_id),
           dist_url="tcp://127.0.0.1:28661",
           num_machines=1,
           machine_rank=0,
           distributed=cfg.distributed,
           args=(cfg,))
