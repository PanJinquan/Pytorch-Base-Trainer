# -*- coding:utf-8 -*-
import os
import argparse
import numbers
import easydict
import yaml
import copy
import json
from pybaseutils import file_utils


def parser_work_space(work_dir, flags: list = [], time=True):
    """生成工程空间
    flag = [cfg.net_type, cfg.width_mult, cfg.input_size[0], cfg.input_size[1],
            cfg.loss_type, cfg.optim_type, flag, file_utils.get_time()]
    """
    if isinstance(flags, str):
        flags = [flags]
    if time:
        flags += [file_utils.get_time()]
    name = [str(n) for n in flags if n]
    name = "_".join(name)
    work_dir = os.path.join(work_dir, name)
    return work_dir


def parser_config(args: argparse.Namespace, cfg_updata: bool = True):
    """
    解析并合并配置参数：(1)命令行argparse (2)使用*.yaml配置文件
    :param args: 命令行参数
    :param cfg_updata:True: 合并配置参数时，相同参数由*.yaml文件参数决定
                     False: 合并配置参数时，相同参数由命令行argparse参数决定
    :return:
    """
    if "config_file" in args and args.config_file:
        cfg = load_config(args.config_file)
        if cfg_updata:
            cfg = update_dict(args.__dict__, cfg)
        else:
            cfg = update_dict(cfg, args.__dict__)
        cfg["config_file"] = args.config_file
    else:
        cfg = args.__dict__
        cfg['config_file'] = save_config(cfg, 'args_config.yaml')
    print_dict(cfg)
    cfg = easydict.EasyDict(cfg)
    return cfg


def easy2dict(config: easydict.EasyDict):
    """
    :param config: EasyDict参数
    """
    # fix a Bug: cfg = dict(config) 仅仅转换第一层easydict
    cfg = json.loads(json.dumps(config))
    return cfg


def dict2easy(config: dict):
    """
    :param config: Dict参数
    """
    return easydict.EasyDict(config)


class Dict2Obj:
    '''
    dict转类对象
    '''

    def __init__(self, args):
        self.__dict__.update(args)


def parser_config_file(config: easydict.EasyDict, config_file: str, cfg_updata: bool = True):
    """
    解析并合并配置参数
    :param config: EasyDict参数
    :param cfg_updata:True: 合并配置参数时，相同参数由config参数决定
                     False: 合并配置参数时，相同参数由config_file中的参数决定
    :return:
    """
    cfg = load_config(config_file)
    if cfg_updata:
        cfg = update_dict(cfg, easy2dict(config))
    else:
        cfg = update_dict(easy2dict(config), cfg)
    print_dict(cfg)
    cfg = easydict.EasyDict(cfg)
    return cfg


def parser_config_name(config: dict or str, key="", join="_"):
    """
    :param config: config
    :param key:
    :param join:拼接方式，为空返回list
    :return: List or String
    """
    if isinstance(config, dict) or isinstance(config, easydict.EasyDict):
        cfg = copy.deepcopy(config)
        if key:
            cfg = config[key].copy()
        if isinstance(cfg, dict):
            cfg = list(cfg.keys())
        if not isinstance(cfg, list):
            cfg = [cfg]
        assert isinstance(cfg, list)
        cfg = [str(c) for c in cfg if c]
        if join:
            cfg = "{}".format(join).join(cfg)
    else:
        cfg = config
    return cfg


def update_dict(cfg1: dict, cfg2: dict):
    """相同参数由后面的cfg2决定"""
    # 不同参数进行合并,相同参数由cfg2替换(如果参数是dict类型，会被cfg2整体替换)
    cfg = dict(cfg1, **cfg2)
    # 查找相同参数，cfg1中被cfg2替换，但没有更新的参数
    for k1, v1 in cfg1.items():
        if isinstance(v1, dict):
            cfg[k1] = update_dict(cfg1[k1], cfg2[k1]) if k1 in cfg2 else v1
    return cfg


def load_config(config_file='config.yaml'):
    """
    读取配置文件，并返回一个python dict 对象
    :param config_file: 配置文件路径
    :return: python dict 对象
    """
    with open(config_file, 'r', encoding="UTF-8") as stream:
        try:
            config = yaml.load(stream, Loader=yaml.FullLoader)
            # config = Dict2Obj(config)
        except yaml.YAMLError as e:
            print(e)
            return None
    return config


def save_config(cfg: dict, config_file='config.yaml'):
    """保存yaml文件"""
    if isinstance(cfg, easydict.EasyDict) or isinstance(cfg, argparse.Namespace):
        cfg = cfg.__dict__
    fw = open(config_file, 'w', encoding='utf-8')
    yaml.dump(cfg, fw)
    return config_file


def print_dict(dict_data, save_path=None):
    list_config = []
    print("=" * 60)
    for key in dict_data:
        info = "{}: {}".format(key, dict_data[key])
        print(info)
        list_config.append(info)
    if save_path is not None:
        with open(save_path, "w") as f:
            for info in list_config:
                f.writelines(info + "\n")
    print("=" * 60)


if __name__ == '__main__':
    data = None
    # config_file = "config.yaml"
    # save_config(data, config_file)
    cfg1 = {"A": 0, "B": {"X": 0, "Y": 0}, "C": {"ADCD"}}
    cfg2 = {"A": 1, "B": {"X": 1, "Z": {"A": 1}}, "D": 1}
    cfg = update_dict(cfg1, cfg2)
    cfg = easydict.EasyDict(cfg)
    print(cfg)
