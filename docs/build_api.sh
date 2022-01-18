#!/usr/bin/env bash
# 参考：https://xiulian.blog.csdn.net/article/details/83657029
# 1.使用sphinx-quickstart在source默认生成conf.py,index.rst等文件
# sphinx-quickstart

# 2.sphinx-apidoc生成接口文档rst
sphinx-apidoc -o source ../basetrainer/

# 3.清理文件
make clean

# 4.生成html文件
make html

# 5.打开build/html/index.html