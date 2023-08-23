# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-12 20:13:04
"""
import os
import basetrainer
import pypandoc
from setuptools import setup, find_packages

root = os.path.dirname(__file__)
# long_description = pypandoc.convert_file(source_file='README.md', to='rst')
long_description =""
setup(name='basetrainer',
      version=basetrainer.__version__,
      description='basetrainer',
      long_description=long_description,
      url='https://github.com/PanJinquan/Pytorch-Base-Trainer',
      author='PanJinquan',
      author_email='pan_jinquan@163.com',
      license='MIT',
      packages=find_packages(where=".",
                             exclude=["build", "configs", "data", "dist", "basetrainer.egg-info", "test",
                                      '.idea', ".gitignore",
                                      ]),  # 为空为全部
      package_data={
          # 如果包中含有.txt文件，则包含它
          'pjq': ['*.jpg'],
          # 包含demo包data文件夹中的 *.dat文件
          'demo': ['data/*.dat']
      },
      install_requires=[
          # 'pandas>=0.20.0',  # 所需要包的版本号
          # 'numpy>=1.14.0'  # 所需要包的版本号
      ],
      zip_safe=False)


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


# readme2rst()
