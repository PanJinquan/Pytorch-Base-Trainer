# MNN

- https://www.mnn.zone
- 教程： https://mnn-docs.readthedocs.io/en/latest/inference/python.html
- 教程： https://www.yuque.com/mnn/en/usage_in_python

# 安装

MNN 的后端支持（如 CUDA、OpenCL）不是默认启用的，必须在编译 MNN 时显式开启。
如果你使用的是预编译的 pip 安装包（如 pip install MNN），官方 PyPI 包通常只包含 CPU 后端，不包含 GPU 支持
安装方法：https://github.com/alibaba/MNN/blob/master/pymnn/INSTALL.md
python3 build_deps.py opencl vulkan
python3 setup.py install # 安装当前环境中
python3 setup.py bdist_wheel # 构建 wheel