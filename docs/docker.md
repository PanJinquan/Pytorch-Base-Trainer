# Docker
```bash
image="glenaaa/tensorrt-ubuntu20.04-cuda12.1:v0.6" # python3.11.5 测试可用
docker build -t=$image .
docker run -it --gpus all -p 40000:80 -v `pwd`:/workspace $image /bin/bash
```