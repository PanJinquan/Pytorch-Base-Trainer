#FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
#FROM docker.dm-ai.cn/algorithm-research/ubuntu22.04-cuda12.1-cudnn8-py310:opencv
FROM docker.dm-ai.cn/algorithm-research/ubuntu22.04-cuda12.1-cudnn8-py310:torch2.8
#FROM docker.dm-ai.cn/algorithm-research/ubuntu22.04-cuda12.1-cudnn8-py310:torch2.8-tensorrt8.6

WORKDIR /app
ADD app /app/
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pybaseutils
RUN pip install easydict
RUN pip install torch==2.5.0 torchvision

CMD ["python", "main.py"]
