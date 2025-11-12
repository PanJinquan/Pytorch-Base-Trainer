# RKNN 模型

## 性能测试

|     | model                       | size             | quant | device | Time/ms   | describe |
|-----|-----------------------------|------------------|-------|--------|-----------|----------|
| MNN | data/model/yolov8n-seg.onnx | [1, 3, 640, 640] | 0     | cpu    | 141.057ms |          |
| MNN | data/model/yolov8n-seg.onnx | [1, 3, 640, 640] | 1     | cpu    | 105.057ms | 量化加速明显   |
