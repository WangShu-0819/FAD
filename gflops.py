import torch, thop

# 硬件加速验证
print(f"CUDA可用性: {torch.cuda.is_available()}")
print(f"当前显卡: {torch.cuda.get_device_name(0)}")

# 计算图分析能力测试
model = torch.nn.Conv2d(3, 64, kernel_size=3)
input = torch.randn(1, 3, 224, 224)
macs, params = thop.profile(model, inputs=(input,))
print(f"标准卷积层计算量: {macs/1e9:.2f} GFLOPs")