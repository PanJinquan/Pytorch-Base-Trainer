FROM glenaaa/tensorrt-ubuntu20.04-cuda12.1:v0.6

WORKDIR /app
ADD app /app/
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pybaseutils
RUN pip install easydict
RUN pip install torch==2.5.0 torchvision

CMD ["python", "main.py"]
