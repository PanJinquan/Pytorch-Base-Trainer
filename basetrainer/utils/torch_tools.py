# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-06-02 16:00:47
# --------------------------------------------------------
"""
import torch
import random
import os
import numpy as np
from collections import OrderedDict
from collections.abc import Iterable


def get_torch_version():
    try:
        v = torch.__version__
        print("torch.version:{}".format(v))
        vid = torch_version_id(v)
    except Exception as e:
        vid = None
    return vid


def torch_version_id(v: str):
    vid = v.split(".")
    vid = float("{}.{:0=2d}".format(vid[0], int(vid[1])))
    return vid


def set_env_random_seed(seed=2020):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def get_loacl_eth():
    '''
    想要获取linux设备网卡接口，并用列表进行保存
    :return:
    '''
    eth_list = []
    cmd = "ls -l /sys/class/net/ | grep -v virtual | sed '1d' | awk 'BEGIN {FS=\"/\"} {print $NF}'"
    try:
        with os.popen(cmd) as f:
            for line in f.readlines():
                line = line.strip()
                eth_list.append(line.lower())
    except Exception as e:
        print(e, "can not found eth,will set default eth is:eth0")
        eth_list = ["eth0"]
    if not eth_list:
        eth_list = ["eth0"]
    return eth_list


def set_node_env(master_addr="localhost", master_port="1200", eth_name=None):
    """
    设置多卡训练的节点信息
    parser = argparse.ArgumentParser(description="for face verification train")
    parser.add_argument("-c", "--config", help="configs file", default="configs/config_distributed.yaml", type=str)
    parser.add_argument("-e", '--eth_name', type=str, default=None, help="set eth name")
    parser.add_argument("-a", '--master_addr', type=str, default='localhost', help="set master node address")
    parser.add_argument("-p", '--master_port', type=str, default='1200', help="set master node port")
    parser.add_argument("--local_rank", type=int, default=0, help="torch.distributed.launch会给模型分配一个args.local_rank的参数，"
                                                                  "也可以通过torch.distributed.get_rank()获取进程id")
    parser.add_argument("--init_method", type=str, default="env://")
    args = parser.parse_args()
    ====================================
    :param master_addr: 主节点地址,default localhost
    :param master_port: 主节点接口,default 1200
    :param eth_name: 网卡名称，None会自动获取
    :return:
    """
    if eth_name is None:  # auto get eth_name
        eth_name = get_loacl_eth()[0]
        print("eth_name:{}".format(eth_name))
    os.environ['NCCL_SOCKET_IFNAME'] = eth_name
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port


def set_distributed_env(backend="nccl", init_method="env://"):
    """
    initialize the distributed environment
    :param backend:
    :param init_method:
    :return: world_size :参与工作的进程数
             rank: 当前进程的rank(这个Worker是全局第几个Worker)
             local_rank：这个Worker是这台机器上的第几个Worker
    """
    # initialize process group
    # use nccl backend to speedup gpu communication
    torch.distributed.init_process_group(backend=backend, init_method=init_method)
    world_size = torch.distributed.get_world_size()
    # torch.distributed.launch 会给模型分配一个args.local_rank的参数，也可以通过torch.distributed.get_rank()获取进程id。
    rank = torch.distributed.get_rank()  # os.environ["RANK"]
    return world_size, rank


def get_distributed_sampler(dataset: torch.utils.data.Dataset, world_size, rank):
    """
    Example:
        sampler = DistributedSampler(dataset) if is_distributed else None
        loader = DataLoader(dataset, shuffle=(sampler is None),
                             sampler=sampler)
        for epoch in range(start_epoch, n_epochs):
             if is_distributed:
                 sampler.set_epoch(epoch)
             train(loader)
    :param dataset:
    :param world_size:
    :param rank:
    :return:
    """
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)
    return sampler


def get_device():
    """
    返回当前设备索引
    torch.cuda.current_device()
    返回GPU的数量
    torch.cuda.device_count()
    返回gpu名字，设备索引默认从0开始
    torch.cuda.get_device_name(0)
    cuda是否可用
    torch.cuda.is_available()
    ==========
    CUDA_VISIBLE_DEVICES=4,5,6 python train.py

    Usage:
    gpu_id = get_device()
    model = build_model()
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=gpu_id)
    ...
    :return:
    """
    gpu_id = list(range(torch.cuda.device_count()))
    return gpu_id


def print_model(model):
    """
    :param model:
    :return:
    """
    for k, v in model.named_parameters():
        # print(k,v)
        print(k)


def freeze_net_layers(net):
    """
    https://www.zhihu.com/question/311095447/answer/589307812
    example:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
    :param net:
    :return:
    """
    # for param in net.parameters():
    #     param.requires_grad = False
    for name, child in net.named_children():
        # print(name, child)
        for param in child.parameters():
            param.requires_grad = False


def rename_module(state_dict, name_map={}):
    """
    重新映射模块名称
    :param state_dict: model state_dict
    :param name_map: 需要重新命名的模型， name_map = {"base_net": "backbone"},即将base_net重新命名为backbone
    :return:
    """
    # 初始化一个空 dict
    new_state_dict = OrderedDict()
    # 修改 key，没有module字段则需要不del，如果有，则需要修改为 module.features
    for k, v in state_dict.items():
        for key, value in name_map.items():
            if key in k: k = k.replace(key, value)
        new_state_dict[k] = v
    return new_state_dict


def load_state_dict(model_path):
    """
    Usage:
        model=Model()
        state_dict = torch_tools.load_state_dict(model_path, module=False)
        model.load_state_dict(state_dict)
    :param model_path:
    :return: state_dict
    """
    state_dict = None
    if model_path:
        print('=> loading model from {}'.format(model_path))
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if 'module' in list(state_dict.keys())[0]:
            state_dict = get_module_state_dict(state_dict)
    else:
        raise Exception("Error:no model file:{}".format(model_path))
    return state_dict


def get_module_state_dict(state_dict):
    """
    :param state_dict:
    :return:
    """
    # 初始化一个空 dict
    new_state_dict = OrderedDict()
    # 修改 key，没有module字段则需要不del，如果有，则需要修改为 module.features
    for k, v in state_dict.items():
        if k.startswith("module."):
            # k = k.replace('module.', '')
            k = k[len("module."):]
        new_state_dict[k] = v
    return new_state_dict


def load_pretrained_model(model, ckpt):
    """Loads pretrianed weights to model.
    Features:只会加载完全匹配的模型参数，不匹配的模型将会忽略
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".
    Args:
        model (nn.Module): network model.
        ckpt (str): OrderedDict or model file
    """
    if isinstance(ckpt, str):
        checkpoint = load_state_dict(ckpt)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    elif isinstance(ckpt, OrderedDict):
        state_dict = ckpt
    else:
        raise Exception("nonsupport type:{} ".format(ckpt))
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print("=" * 60)
    if len(matched_layers) == 0:
        print('Warning:The model checkpoint cannot be loaded,'
              'please check the key names manually')
    else:
        print('Successfully loaded model checkpoint')
        # [print('{}'.format(layer)) for layer in matched_layers]
        if len(discarded_layers) > 0:
            print('The following layers are discarded due to unmatched keys or layer size')
            [print('{}'.format(layer)) for layer in discarded_layers]

    print("=" * 60)
    return model


def plot_model(model, output=None, input_shape=None):
    """
    Usage:
    output = model(inputs)
    vis_graph = make_dot(output, params=dict(model.named_parameters()))
    vis_graph.view()
    =================================================================
    output/input_shape至少已知一个
    :param model:
    :param output:
    :param input_shape: (batch_size, 3, input_size[0], input_size[1])
    :return:
    """
    from torchviz import make_dot
    if output is None:
        output = model_forward(model, input_shape, device="cpu")
    vis_graph = make_dot(output, params=dict(model.named_parameters()))
    vis_graph.view()


def model_forward(model, input_shape, device="cpu"):
    """
    input_shape=(batch_size, 3, input_size[0], input_size[1])
    :param model:
    :param input_shape:
    :param device:
    :return:
    """
    inputs = torch.randn(size=input_shape)
    inputs = inputs.to(device)
    model = model.to(device)
    model.eval()
    output = model(inputs)
    return output


def summary_model(model, batch_size=1, input_size=[112, 112], plot=False, device="cpu"):
    """
    ----This tools can show----
    Total params: 359,592
    Total memory: 47.32MB
    Total MAdd: 297.37MMAdd
    Total Flops: 153.31MFlops
    Total MemR+W: 99.7MB
    ====================================================
    https://www.cnblogs.com/xuanyuyt/p/12653041.html
    Total number of network parameters (params)
    Theoretical amount of floating point arithmetics (FLOPs)
    Theoretical amount of multiply-adds (MAdd MACC) (乘加运算)
    Memory usage (memory)
    MACCs：是multiply-accumulate operations，指点积运算， 一个 macc = 2FLOPs
    FLOPs 的全称是 floating points of operations，即浮点运算次数，用来衡量模型的计算复杂度。
    计算 FLOPs 实际上是计算模型中乘法和加法的运算次数。
    卷积层的浮点运算次数不仅取决于卷积核的大小和输入输出通道数，还取决于特征图的大小；
    而全连接层的浮点运算次数和参数量是相同的。
    ====================================================
    :param model:
    :param batch_size:
    :param input_size: (W,H) or (B, C, H, W)
    :param plot: plot model
    :param device:
    :return:
    """
    from torchsummary import summary
    from torchstat import stat
    if len(input_size) == 2:
        shape = (batch_size, 3, input_size[1], input_size[0])
    elif len(input_size) == 4:
        shape = tuple(input_size)
    else:
        raise Exception("input_size error:{}".format(input_size))
    B, C, H, W = shape
    # inputs = torch.randn(size=(B, 3, input_size[1], input_size[0]))
    inputs = torch.randn(size=shape)
    inputs = inputs.to(device)
    model = model.to(device)
    model.eval()
    output = model(inputs)
    # 统计模型参数
    summary(model, input_size=(C, H, W), batch_size=B, device=device)
    # 统计模型参数和计算FLOPs
    stat(model, (C, H, W))
    # summary可能报错，可使用该方法
    # summary_v2(model, inputs, item_length=26, verbose=True)
    # from thop import profile
    # macs, params = profile(model, inputs=(inputs,))
    # print("Total Flops :{}".format(macs))
    # print("Total params:{}".format(params))
    print("===" * 10)
    print("inputs.shape:{}".format(inputs.shape))
    # print("output.shape:{}".format(output.shape))
    if plot:
        plot_model(model, output)


def nni_summary_model(model, batch_size=1, input_size=[112, 112], plot=False, device="cpu"):
    """
    https://nni.readthedocs.io/zh/stable/Compression/CompressionUtils.html
    NNI 提供了模型计数器，用于计算模型的 FLOPs 和参数。
    此计数器支持计算没有掩码模型的 FLOPs、参数，也可以计算有掩码模型的 FLOPs、参数，
    这有助于在模型压缩过程中检查模型的复杂度。
    注意，对于结构化的剪枝，仅根据掩码来标识保留的滤波器，不会考虑剪枝的输入通道，
    因此，计算出的 FLOPs 会比实际数值要大（即，模型加速后的计算值）。
    我们支持两种模式来收集模块信息。 第一种是 default 模式，它只采集卷积操作和线性操作的信息。
    第二种是 full 模式，它还会收集其他操作的信息。 用户可以轻松地使用我们收集的 results 进行进一步的分析。
    :param model:
    :param batch_size:
    :param input_size: (W,H) or (B, C, H, W)
    :param plot:
    :param device:
    :return:
    """
    # pip install nni==2.0
    # from nni.compression.utils.counter import count_flops_params
    from nni.compression.pytorch.utils.counter import count_flops_params
    if len(input_size) == 2:
        shape = (batch_size, 3, input_size[1], input_size[0])
    elif len(input_size) == 4:
        shape = tuple(input_size)
    else:
        raise Exception("input_size error:{}".format(input_size))
    B, C, H, W = shape
    inputs = torch.randn(size=shape)
    inputs = inputs.to(device)
    model = model.to(device)
    model.eval()
    output = model(inputs)
    flops, params, _ = count_flops_params(model, inputs, verbose=True)
    info = f"Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M"
    print(info)
    if plot:
        plot_model(model, output)


def torchinfo_summary(model, batch_size=1, input_size=[112, 112], plot=False, device="cpu"):
    """
    ----This tools can show----
    Total params: 359,592
    Total memory: 47.32MB
    Total MAdd: 297.37MMAdd
    Total Flops: 153.31MFlops
    Total MemR+W: 99.7MB
    ====================================================
    https://www.cnblogs.com/xuanyuyt/p/12653041.html
    Total number of network parameters (params)
    Theoretical amount of floating point arithmetics (FLOPs)
    Theoretical amount of multiply-adds (MAdd MACC) (乘加运算)
    Memory usage (memory)
    MACCs：是multiply-accumulate operations，指点积运算， 一个 macc = 2FLOPs
    FLOPs 的全称是 floating points of operations，即浮点运算次数，用来衡量模型的计算复杂度。
    计算 FLOPs 实际上是计算模型中乘法和加法的运算次数。
    卷积层的浮点运算次数不仅取决于卷积核的大小和输入输出通道数，还取决于特征图的大小；
    而全连接层的浮点运算次数和参数量是相同的。
    ====================================================
    :param model:
    :param batch_size:
    :param input_size: (W,H) or (B, C, H, W)
    :param plot: plot model
    :param device:
    :return:
    """
    from torchinfo import summary
    from torchstat import stat
    if len(input_size) == 2:
        shape = (batch_size, 3, input_size[1], input_size[0])
    elif len(input_size) == 4:
        shape = tuple(input_size)
    else:
        raise Exception("input_size error:{}".format(input_size))
    B, C, H, W = shape
    inputs = torch.randn(size=shape)
    inputs = inputs.to(device)
    model = model.to(device)
    model.eval()
    output = model(inputs)
    # 统计模型参数
    summary(model, input_size=shape, device=device)
    # 统计模型参数和计算FLOPs
    stat(model, (C, H, W))
    # summary可能报错，可使用该方法
    # summary_v2(model, inputs, item_length=26, verbose=True)
    # from thop import profile
    # macs, params = profile(model, inputs=(inputs,))
    # print("Total Flops :{}".format(macs))
    # print("Total params:{}".format(params))
    print("===" * 10)
    print("inputs.shape:{}".format(inputs.shape))
    # print("output.shape:{}".format(output.shape))
    if plot:
        plot_model(model, output)


def print_model_shape(inp, out):
    """
    打印模型输入输出维度
    :param inp:
    :param out:
    :return:
    """
    if isinstance(inp, torch.Tensor): inp = [inp]
    if isinstance(out, torch.Tensor): out = [out]
    print("===" * 10)
    for i in range(len(inp)):
        print("input{}  {}".format(i, inp[i].shape))
    for i in range(len(out)):
        print("output{} {}".format(i, out[i].shape))
    print("===" * 10)
