# RKNN 模型

#### RKNN安装(RK3588)

- https://github.com/airockchip/rknn-toolkit2
- 详细看`02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_CN.pdf`
- PC端：pip install docs/rknn-toolkit2/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
- 板端：pip install
  docs/rknn-toolkit2/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
- 板端更新npu：

```bash
cd /home/youyeetoo/nasdata/release/infrastructure/
# 查询rknn_server版本
strings /usr/bin/rknn_server | grep -i "rknn_server version"
# 显示rknn_server版本为X.X.X
# rknn_server version: X.X.X
# 查询librknnrt.so库版本
strings /usr/lib/librknnrt.so | grep -i "librknnrt version"
# 显示librknnrt库版本为X.X.X
# librknnrt version: X.X.X
# 更新RKNN Server服务和librknnrt.so库
ps aux | grep python # 查询python进程
ps aux | grep rknn_server # 查询rknn_server进程
kill -9 `pgrep rknn_server` # 关闭当前RKNN Server服务进程。
sudo systemctl stop rknn_server #  停止
sudo systemctl disable rknn_server 
# 64位库
sudo cp runtime/Linux/rknn_server/aarch64/usr/bin/* /usr/bin
sudo cp runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib64 
sudo cp runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib 
# 32位库
sudo cp runtime/Linux/rknn_server/armhf/usr/bin/* /usr/bin
sudo cp runtime/Linux/librknn_api/armhf/librknnrt.so /usr/lib

#重启RKNN Server服务：
sudo chmod +x /usr/bin/rknn_server
sudo nohup /usr/bin/rknn_server >/dev/null&
sudo restart_rknn.sh
```

## RKNN量化

- 详细看`02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_CN.pdf`《7.1 模拟器精度排查》
  RKNPU目前不支持FP32的计算方式，因此模拟器在不开启量化的情况下，默认是FP16的运算类型，所以 只需要在使用rknn.build()
  接口时，将do_quantization参数设置为False，即可以将原始模型转换为FP16的
  RKNN模型，接着调用rknn.init_runtime(target=None)和rknn.inference()接口进行FP16模拟推理并获取输出结果

## 性能测试

|      | model                       | FLOPs | size             | quant | device | Time/ms | describe |
|------|-----------------------------|-------|------------------|-------|--------|---------|----------|
| MNN  | data/model/yolov8n-seg.onnx | 12.1G | [1, 3, 640, 640] | 0     | cpu    | 141     |          |
| MNN  | data/model/yolov8n-seg.onnx | 12.1G | [1, 3, 640, 640] | 1     | cpu    | 105     | 量化加速明显   |
| RKNN | data/model/yolov8n-seg.rknn | 12.1G | [1, 3, 640, 640] | 0     | npu    | 95      |          |
| RKNN | data/model/yolov8n-seg.rknn | 12.1G | [1, 3, 480, 480] | 0     | npu    | 63      |          |
| RKNN | data/model/yolov8n-seg.rknn | 12.1G | [1, 3, 320, 320] | 0     | npu    | 24      |          |
