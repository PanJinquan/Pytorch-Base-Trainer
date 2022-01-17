#!/usr/bin/env bash
# 制作pip包： https://www.cnblogs.com/sting2me/p/6550897.html
# 发布pip包： https://packaging.python.org/tutorials/packaging-projects/
rm -rf dist/*
python setup.py sdist
python setup.py bdist_wheel --universal
pip install dist/basetrainer-*.*.*.tar.gz


