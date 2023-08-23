Pytorch-Base-Trainer(PBT)
=========================

-  开源不易,麻烦给个【Star】
-  Github: https://github.com/PanJinquan/Pytorch-Base-Trainer
-  pip安装包： https://pypi.org/project/basetrainer/
-  博客地址：https://panjinquan.blog.csdn.net/article/details/122662902

1.Introduction
--------------

考虑到深度学习训练过程都有一套约定成俗的流程，鄙人借鉴\ **Keras**\ 开发了一套基础训练库：
**Pytorch-Base-Trainer(PBT)**\ ；
这是一个基于Pytorch开发的基础训练库，支持以下特征：

-  [x]
   支持多卡训练训练(DP模式)和分布式多卡训练(DDP模式)，参考\ `build_model_parallel <basetrainer/utils/torch_data.py>`__
-  [x]
   支持argparse命令行指定参数，也支持\ `config.yaml <configs/config.yaml>`__\ 配置文件
-  [x]
   支持最优模型保存\ `ModelCheckpoint <basetrainer/callbacks/model_checkpoint.py>`__
-  [x]
   支持自定义回调函数\ `Callback <basetrainer/callbacks/callbacks.py>`__
-  [x] 支持NNI模型剪枝(\ **L1/L2-Pruner,FPGM-Pruner
   Slim-Pruner**)\ `nni_pruning <basetrainer/pruning/nni_pruning.py>`__
-  [x] 非常轻便,安装简单

诚然，诸多大公司已经开源基础库，如MMClassification,MMDetection等库；
但碍于这些开源库安装麻烦,依赖库多,版本差异大等问题；鄙人开发了一套比较基础的训练Pipeline：
**Pytorch-Base-Trainer(PBT)**, 基于PBT可以快速搭建自己的训练工程；
目前，基于PBT完成了\ **通用分类库(PBTClassification),通用检测库(PBTDetection),通用语义分割库(
PBTSegmentation)以及,通用姿态检测库(PBTPose)**

+-----------------------+-----------------------+-----------------------+
| **通用库**            | **类型**              | **说明**              |
+=======================+=======================+=======================+
| **PBTClassification** | 通用分类库            | 集成常用的分类模型，支持多种数据格式,样本 |
|                       |                       | 重采样                |
+-----------------------+-----------------------+-----------------------+
| **PBTDetection**      | 通用检测库            | 集成常用的检测类模型，如RFB,SSD和Y |
|                       |                       | OLOX                  |
+-----------------------+-----------------------+-----------------------+
| **PBTSegmentation**   | 通用语义分割库        | 集成常用的语义分割模型，如DeepLab, |
|                       |                       | UNet等                |
+-----------------------+-----------------------+-----------------------+
| **PBTPose**           | 通用姿态检测库        | 集成常用的人体姿态估计模型,如UDP,Si |
|                       |                       | mple-base-line        |
+-----------------------+-----------------------+-----------------------+

基于PBT框架训练的模型,已经形成了一套完整的Android端上部署流程,支持CPU和GPU

+-----------------------+-----------------------+-----------------------+
| `人体姿态估计2DPose <https: | `人脸+人体检测 <https://blo | `人像抠图 <https://blog.c |
| //blog.csdn.net/guyue | g.csdn.net/guyuealian | sdn.net/guyuealian/ar |
| alian/article/details | /article/details/1206 | ticle/details/1216809 |
| /115765863>`__        | 88804>`__             | 39>`__                |
+=======================+=======================+=======================+
|                       |                       |                       |
+-----------------------+-----------------------+-----------------------+
| CPU/GPU:70/50ms       | CPU/GPU:30/20ms       | CPU/GPU:150/30ms      |
+-----------------------+-----------------------+-----------------------+

..

   PS：受商业保护,目前,仅开源Pytorch-Base-Trainer(PBT),基于PBT的分类,检测和分割以及姿态估计训练库,暂不开源。

2.Install
---------

-  源码安装

.. code:: bash

   git clone https://github.com/PanJinquan/Pytorch-Base-Trainer
   cd Pytorch-Base-Trainer
   bash setup.sh #pip install dist/basetrainer-*.*.*.tar.gz

-  pip安装: https://pypi.org/project/basetrainer/

.. code:: bash

   # 安装方法1:(有延时，可能不是最新版本)
   pip install basetrainer 
   # 安装方法2:(从pypi源下载最新版本)
   pip install --upgrade basetrainer -i https://pypi.org/simple

-  使用\ `NNI <https://github.com/microsoft/nni>`__
   模型剪枝工具，需要安装\ `NNI <https://github.com/microsoft/nni>`__

.. code:: bash

   # Linux or macOS
   python3 -m pip install --upgrade nni
   # Windows
   python -m pip install --upgrade nni

3.训练框架
----------

PBT基础训练库定义了一个基类(\ `Base <basetrainer/engine/base.py>`__),所有训练引擎(Engine)以及回调函数(Callback)都会继承基类。

(1)训练引擎(Engine)
^^^^^^^^^^^^^^^^^^^

``Engine``\ 类实现了训练/测试的迭代方法(如on_batch_begin,on_batch_end),其迭代过程参考如下,
用户可以根据自己的需要自定义迭代过程：

.. code:: python

   self.on_train_begin()
   for epoch in range(num_epochs):
       self.set_model()  # 设置模型
       # 开始训练
       self.on_epoch_begin()  # 开始每个epoch调用
       for inputs in self.train_dataset:
           self.on_batch_begin()  # 每次迭代开始时回调
           self.run_step()  # 每次迭代返回outputs, losses
           self.on_train_summary()  # 每次迭代，训练结束时回调
           self.on_batch_end()  # 每次迭代结束时回调
       # 开始测试
       self.on_test_begin()
       for inputs in self.test_dataset:
           self.run_step()  # 每次迭代返回outputs, losses
           self.on_test_summary()  # 每次迭代，测试结束时回调
       self.on_test_end()  # 结束测试
       # 结束当前epoch
       self.on_epoch_end()
   self.on_train_end()

``EngineTrainer``\ 类继承\ ``Engine``\ 类,用户需要继承该类,并实现相关接口:

+--------------------+--------------+
| 接口               | 说明         |
+====================+==============+
| build_train_loader | 定义训练数据 |
+--------------------+--------------+
| build_test_loader  | 定义测试数据 |
+--------------------+--------------+
| build_model        | 定义模型     |
+--------------------+--------------+
| build_optimizer    | 定义优化器   |
+--------------------+--------------+
| build_criterion    | 定义损失函数 |
+--------------------+--------------+
| build_callbacks    | 定义回调函数 |
+--------------------+--------------+

另外，\ ``EngineTrainer``\ 类还是实现了两个重要的类方法(build_dataloader和build_model_parallel),用于构建分布式训练

+-----------------------------------+-----------------------------------+
| 类方法                            | 说明                              |
+===================================+===================================+
| build_dataloader                  | 用于构建加载方式,参数distributed设置是否使用分布式加载 |
|                                   | 数据                              |
+-----------------------------------+-----------------------------------+
| build_model_parallel              | 用于构建模型,参数distributed设置是否使用分布式训练模型 |
+-----------------------------------+-----------------------------------+

(2)回调函数(Callback)
^^^^^^^^^^^^^^^^^^^^^

每个回调函数都需要继承(Callback),用户在回调函数中,可实现对迭代方法输入/输出的处理,例如:

+-----------------------------------+-----------------------------------+
| 回调函数                          | 说明                              |
+===================================+===================================+
| `LogHistory <basetrainer/callback | Log历史记录回调函数,可使用Tensorboard可视化 |
| s/log_history.py>`__              |                                   |
+-----------------------------------+-----------------------------------+
| `ModelCheckpoint <basetrainer/cal | 保存模型回调函数,可选择最优模型保存 |
| lbacks/model_checkpoint.py>`__    |                                   |
+-----------------------------------+-----------------------------------+
| `LossesRecorder <basetrainer/call | 单个Loss历史记录回调函数,可计算每个epoch的平均值 |
| backs/losses_recorder.py>`__      |                                   |
+-----------------------------------+-----------------------------------+
| `MultiLossesRecorder <basetrainer | 用于多任务Loss的历史记录回调函数  |
| /callbacks/multi_losses_recorder. |                                   |
| py>`__                            |                                   |
+-----------------------------------+-----------------------------------+
| `AccuracyRecorder <basetrainer/me | 用于计算分类Accuracy回调函数      |
| tric/accuracy_recorder.py>`__     |                                   |
+-----------------------------------+-----------------------------------+
| `get_scheduler <basetrainer/sched | 各种学习率调整策略(MultiStepLR,CosineAnnea |
| uler/build_scheduler.py>`__       | lingLR,ExponentialLR)的回调函数   |
+-----------------------------------+-----------------------------------+

4.使用方法
----------

``basetrainer``\ 使用方法可以参考\ `example.py <./example.py>`__,构建自己的训练器,可通过如下步骤实现：

-  step1:
   新建一个类\ ``ClassificationTrainer``\ ，继承\ ``trainer.EngineTrainer``
-  step2: 实现接口

.. code:: python


   def build_train_loader(self, cfg, **kwargs):
       """定义训练数据"""
       raise NotImplementedError("build_train_loader not implemented!")
   in_file, 'rst', format='md', outputfile="README.rst", encoding='utf-8')

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

-  step3: 在初始化中调用\ ``build``

.. code:: python


   def __init__(self, cfg):
       super(ClassificationTrainer, self).__init__(cfg)
       ...
       self.build(cfg)
       ...

-  step4:
   实例化\ ``ClassificationTrainer``,并使用\ ``launch``\ 启动分布式训练

.. code:: python

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

5.Example
---------

-  ``basetrainer``\ 使用方法可以参考\ `example.py <./example.py>`__

.. code:: bash

   # 单进程多卡训练
   python example.py --gpu_id 0 1 # 使用命令行参数
   python example.py --config_file configs/config.yaml # 使用yaml配置文件
   # 多进程多卡训练(分布式训练)
   python example.py --config_file configs/config.yaml --distributed # 使用yaml配置文件

-  目标支持的backbone有：resnet[18,34,50,101],
   ,mobilenet_v2等，详见\ `backbone <basetrainer/models/build_models.py>`__\ 等
   ，其他backbone可以自定义添加
-  训练参数可以通过两种方法指定: (1) 通过argparse命令行指定
   (2)通过`config.yaml <configs/config.yaml>`__\ 配置文件，当存在同名参数时，以配置文件为默认值

+-----------------+-----------------+-----------------+-----------------+
| **参数**        | **类型**        | **参考值**      | **说明**        |
+=================+=================+=================+=================+
| train_data      | str, list       | -               | 训练数据文件，可支持多个文件 |
+-----------------+-----------------+-----------------+-----------------+
| test_data       | str, list       | -               | 测试数据文件，可支持多个文件 |
+-----------------+-----------------+-----------------+-----------------+
| work_dir        | str             | work_space      | 训练输出工作空间 |
+-----------------+-----------------+-----------------+-----------------+
| net_type        | str             | resnet18        | backbone类型,{res |
|                 |                 |                 | net,resnest,mob |
|                 |                 |                 | ilenet_v2,…}    |
+-----------------+-----------------+-----------------+-----------------+
| input_size      | list            | [128,128]       | 模型输入大小[W,H] |
+-----------------+-----------------+-----------------+-----------------+
| batch_size      | int             | 32              | batch size      |
+-----------------+-----------------+-----------------+-----------------+
| lr              | float           | 0.1             | 初始学习率大小  |
+-----------------+-----------------+-----------------+-----------------+
| optim_type      | str             | SGD             | 优化器，{SGD,Adam} |
+-----------------+-----------------+-----------------+-----------------+
| loss_type       | str             | CELoss          | 损失函数        |
+-----------------+-----------------+-----------------+-----------------+
| scheduler       | str             | multi-step      | 学习率调整策略，{multi- |
|                 |                 |                 | step,cosine}    |
+-----------------+-----------------+-----------------+-----------------+
| milestones      | list            | [30,80,100]     | 降低学习率的节点，仅仅sche |
|                 |                 |                 | duler=multi-ste |
|                 |                 |                 | p有效           |
+-----------------+-----------------+-----------------+-----------------+
| momentum        | float           | 0.9             | SGD动量因子     |
+-----------------+-----------------+-----------------+-----------------+
| num_epochs      | int             | 120             | 循环训练的次数  |
+-----------------+-----------------+-----------------+-----------------+
| num_warn_up     | int             | 3               | warn_up的次数   |
+-----------------+-----------------+-----------------+-----------------+
| num_workers     | int             | 12              | DataLoader开启线程数 |
+-----------------+-----------------+-----------------+-----------------+
| weight_decay    | float           | 5e-4            | 权重衰减系数    |
+-----------------+-----------------+-----------------+-----------------+
| gpu_id          | list            | [ 0 ]           | 指定训练的GPU卡号，可指定多 |
|                 |                 |                 | 个              |
+-----------------+-----------------+-----------------+-----------------+
| log_freq        | in              | 20              | 显示LOG信息的频率 |
+-----------------+-----------------+-----------------+-----------------+
| finetune        | str             | model.pth       | finetune的模型  |
+-----------------+-----------------+-----------------+-----------------+
| use_prune       | bool            | True            | 是否进行模型剪枝 |
+-----------------+-----------------+-----------------+-----------------+
| progress        | bool            | True            | 是否显示进度条  |
+-----------------+-----------------+-----------------+-----------------+
| distributed     | bool            | False           | 是否使用分布式训练 |
+-----------------+-----------------+-----------------+-----------------+

-  学习率调整策略

+---------------+------------------------+--------------------+
| **scheduler** | **说明**               | **lr-epoch曲线图** |
+===============+========================+====================+
| multi_step    | 阶梯学习率调整策略     |                    |
+---------------+------------------------+--------------------+
| cosine        | 余弦退火学习率调整策略 |                    |
+---------------+------------------------+--------------------+
| ExpLR         | 指数衰减学习率调整策略 |                    |
+---------------+------------------------+--------------------+
| LambdaLR      | Lambda学习率调整策略   |                    |
+---------------+------------------------+--------------------+

6.可视化
--------

目前训练过程可视化工具是使用Tensorboard，使用方法：

.. code:: bash

   tensorboard --logdir=path/to/log/

+--+--+
|  |  |
+==+==+
|  |  |
+--+--+
|  |  |
+--+--+

7.其他
------

+----------+---------------------+
| 作者     | PKing               |
+==========+=====================+
| 联系方式 | pan_jinquan@163.com |
+----------+---------------------+
