# Pytorch-Base-Trainer(PBT)

## 1.Introduction

考虑到深度学习训练过程都有一套约定成俗的流程，鄙人借鉴**Keras**开发了一套基础训练库： **Pytorch-Base-Trainer(PBT)**； 这是一个基于Pytorch开发的基础训练库，支持以下特征：

- [x] 支持多卡训练训练(DP模式)和分布式多卡(DDP模式)，参考[build_model_parallel](basetrainer/utils/torch_data.py)
- [x] 支持argparse命令行指定参数，也支持[config.yaml](configs/config.yaml)配置文件
- [x] 支持最优模型保存[ModelCheckpoint](basetrainer/callbacks/model_checkpoint.py)
- [x] 支持自定义回调函数[Callback](basetrainer/callbacks/callbacks.py)
- [x] 支持NNI模型剪枝(**L1/L2-Pruner,FPGM-Pruner Slim-Pruner**)[nni_pruning](basetrainer/pruning/nni_pruning.py)
- [x] 非常轻便,安装简单

诚然，诸多大公司已经开源基础库，如MMClassification,MMDetection等库； 但碍于这些开源库安装麻烦,依赖库多,版本差异大等问题；鄙人还是开发了一套属于自己的， 比较lowbi的基础训练库**
Pytorch-Base-Trainer(PBT)**, 基于PBT可以快速搭建自己的训练工程； 目前，基于PBT完成了**通用分类库(PBTClassification),通用检测库(PBTDetection),通用语义分割库(
PBTSegmentation)以及,通用姿态检测库(PBTPose)**

|**通用库**              |**类型**          |**说明**                                           |
|:-----------------------|:-----------------|:--------------------------------------------------|
|**PBTClassification**   |通用分类库        | 集成常用的分类模型，支持多种数据格式,样本重采样   |
|**PBTDetection**        |通用检测库        | 集成常用的检测类模型，如RFB,SSD和YOLOX            |
|**PBTSegmentation**     |通用语义分割库    | 集成常用的语义分割模型，如DeepLab,UNet等          |
|**PBTPose**             |通用姿态检测库    | 集成常用的人体姿态估计模型,如UDP,Simple-base-line |

<img src="docs/source/basetrainer.png" width="800" >

## 2.Install

- 源码安装

```bash
git clone https://github.com/PanJinquan/Pytorch-Base-Trainer
cd Pytorch-Base-Trainer
bash setup.sh #pip install dist/basetrainer-*.*.*.tar.gz
```

- pip安装

```bash
pip install basetrainer
```

- 使用[NNI](https://github.com/microsoft/nni) 模型剪枝工具，需要安装[NNI](https://github.com/microsoft/nni)

```bash
# Linux or macOS
python3 -m pip install --upgrade nni
# Windows
python -m pip install --upgrade nni
```

## 3.使用方法

`basetrainer`使用方法可以参考[example.py](./example.py)

- step1: 新建一个类`ClassificationTrainer`，继承`trainer.EngineTrainer`
- step2: 实现接口

```python

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
```

- step3: 在初始化中调用`build`

```python

def __init__(self, cfg):
    super(ClassificationTrainer, self).__init__(cfg)
    ...
    self.build(cfg)
    ...
```

- step4: 实例化`ClassificationTrainer`,并使用`launch`启动分布式训练

```python
def main(cfg):
    t = ClassificationTrainer(cfg)
    return t.run()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cfg = setup_config.parser_config(args)
    launch(main,
           num_gpus_per_machine=len(cfg.gpu_id),
           dist_url="tcp://127.0.0.1:28661",
           num_machines=1,
           machine_rank=0,
           distributed=cfg.distributed,
           args=(cfg,))

```

## 4.回调函数

回调函数需要继承`Callback`, 使用方法可以参考[log_history.py](basetrainer/callbacks/log_history.py)

## 5.Example

- `basetrainer`使用方法可以参考[example.py](./example.py)
- 目标支持的backbone有：resnet[18,34,50,101], ,mobilenet_v2等，详见[backbone](basetrainer/models/build_models.py)等
  ，其他backbone可以自定义添加
- 训练参数可以通过两种方法指定: (1) 通过argparse命令行指定 (2)通过[config.yaml](configs/config.yaml)配置文件，当存在同名参数时，以配置文件为默认值

|**参数**     |**类型**      |**参考值** |**说明**            |
|:----------- |:-------------|:----------|:-------------------|
|train_data   |str, list     |-          |训练数据文件，可支持多个文件|
|test_data    |str, list     |-          |测试数据文件，可支持多个文件|
|class_name   |str,list,dict |-          |需要训练的类别|
|work_dir     |str           |work_space |训练输出工作空间|
|net_type     |str           |resnet18   |backbone类型,{resnet,resnest,mobilenet_v2,...}|
|width_mult   |int           |1.0        |宽度因子|
|flag         |str           |-          |实验标志|
|input_size   |list          |[128,128]  |模型输入大小[W,H]|
|batch_size   |int           |32         |batch size|
|lr           |float         |0.1        |初始学习率大小|
|optim_type   |str           |SGD        |优化器，{SGD,Adam}|
|scheduler    |str           |multi-step |学习率调整策略，{multi-step,cosine}|
|milestones   |list          |[30,80,100]|降低学习率的节点，仅仅scheduler=multi-step有效|
|momentum     |float         |0.9        |SGD动量因子|
|num_epochs   |int           |120        |循环训练的次数|
|num_warn_up  |int           |3          |warn_up的次数|
|num_workers  |int           |12         |DataLoader开启线程数|
|weight_decay |float         |5e-4       |权重衰减系数|
|gpu_id       |list          |[ 0 ]      |指定训练的GPU卡号，可指定多个|
|start_save   |int           |null       |从epochs开始保存模型，null表示仅保存最后10个|
|log_freq     |in            |20         |显示LOG信息的频率|
|pretrained   |bool          |True       |是否使用pretrained|
|finetune     |str           |model.pth  |finetune的模型|
|check        |bool          |True       |是否检测数据，可去除空数据|
|use_prune    |bool          |True       |是否进行模型剪枝|
|progress     |bool          |True       |是否显示进度条|
|distributed  |bool          |False      |是否使用分布式训练|

## 6.可视化

目前训练过程可视化工具是使用Tensorboard，使用方法：

```bash
tensorboard --logdir=path/to/log/
```

|<img src="docs/assets/lr-epoch.png"/>    | <img src="docs/assets/step.png"/>    |
|:--------------------------------------- |:-----------------------------------------|
|<img src="docs/assets/train-acc.png"/>   |<img src="docs/assets/test-acc.png"/>     |
|<img src="docs/assets/train-loss.png"/>  |<img src="docs/assets/test-loss.png"/>    |



## 7.其他说明

@pan_jinquan@163.com