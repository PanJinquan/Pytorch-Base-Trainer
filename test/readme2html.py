# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-02-08 17:37:34
"""
import os
import basetrainer
import pypandoc
from setuptools import setup, find_packages


def readme2rst(in_file='README.md'):
    """
    转化文件的格式。
    convert(source, to, format=None, extra_args=(), encoding='utf-8', outputfile=None, filters=None)
    parameter-
        source：源文件
        to：目标文件的格式，比如html、rst、md等
        format：源文件的格式，比如html、rst、md等。默认为None，则会自动检测
        encoding：指定编码集
        outputfile：目标文件，比如test.html（注意outputfile的后缀要和to一致）
    """
    # 修复中文乱码问题： <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
    pypandoc.convert_file(in_file, 'html', format='md', outputfile="README.html", encoding='utf-8')
    pypandoc.convert_file(in_file, 'rst', format='md', outputfile="README.rst", encoding='utf-8')
    print("OK")


readme2rst()
