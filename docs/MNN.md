# MNN

- https://www.mnn.zone
- 教程： https://mnn-docs.readthedocs.io/en/latest/inference/python.html
- 教程： https://www.yuque.com/mnn/en/usage_in_python

## 依赖
- 在Ubuntu 20.04上从零编译MNN（含Vulkan加速配置） https://blog.csdn.net/weixin_29053383/article/details/159311826
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git libprotobuf-dev protobuf-compiler \
    libvulkan-dev vulkan-utils glslang-tools

#验证Vulkan驱动是否正常工作：
vulkaninfo | grep GPU

```

## 编译MNN

- MNN 的后端支持（如 CUDA、OpenCL）不是默认启用的，必须在编译 MNN 时显式开启。
  如果你使用的是预编译的 pip 安装包（如 pip install MNN），官方 PyPI 包通常只包含 CPU 后端，不包含 GPU 支持
- 安装方法：https://github.com/alibaba/MNN/blob/master/pymnn/INSTALL.md
- 参考`base-utils/docs/opencl.md`和`vulkan.md`安装opencl和vulkan
- 编译MNN时，需要开启opencl和vulkan后端

```bash
# 编译MNN:https://mnn-docs.readthedocs.io/en/latest/compile/pymnn.html
cd MNN/pymnn/pip_package
python3 build_deps.py opencl,vulkan,openmp,llm # 测试正常
python3 setup.py install            # 安装当前环境中
python3 setup.py bdist_wheel        # 构建 wheel
# 安装包在项目根目录下的pymnn_build目录下
pip install  --force-reinstall -U docs/MNN/mnn-3.4.1-cp310-cp310-linux_x86_64.whl  numpy==1.26.0
```

