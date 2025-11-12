#!/usr/bin/env bash
# 制作pip包： https://www.cnblogs.com/sting2me/p/6550897.html
# 发布pip包： https://packaging.python.org/tutorials/packaging-projects/
# sudo apt-get install pandoc
# pip install twine

twine upload dist/*  --verbose
#Enter your username: PKing
#Enter your password:

echo please use PIP to install basetrainer:
echo pip install --upgrade basetrainer -i https://pypi.org/simple
