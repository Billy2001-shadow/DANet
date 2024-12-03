import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F  # 导入 torch.nn.functional
from thop import profile
from options import get_args
from models.net import DANet

args = get_args('test')

# 加载模型并移动到 GPU
model = DANet()

# 打印网络结构和参数量
# summary(model, (3, 424, 564))

# 计算 FLOPs 和参数量
input = torch.randn(1, 3, 224, 308).cuda()
flops, params = profile(model, inputs=(input, ))
print(f"FLOPs: {flops}, Total Params: {params}")

# 计算 Backbone 参数量
backbone_params = sum(p.numel() for p in model.backbone.parameters())
print(f"Backbone Parameters: {backbone_params}")

# 计算 PST 参数量
pst_params = sum(p.numel() for p in model.pst.parameters())
print(f"PST Parameters: {pst_params}")

# 计算 Decoder 参数量
decoder_params = sum(p.numel() for p in model.decoder.parameters())
print(f"Decoder Parameters: {decoder_params}")



# 可视化网络结构
# import torch
# from thop import profile
# from fvcore.nn import FlopCountAnalysis
# from models.net import DANet
# from options import get_args

# args = get_args('test')
# model = DANet(args)
# x = torch.rand(1, 3, 228, 304)  # 输入图像尺寸
# flops, params = profile(model, inputs=(x,))
# print("******************************thop**********************************")
# print(f"FLOPs: {flops}")  # 输出 FLOPs
# print(f"Parameters: {params}")  # 输出参数量
# print("******************************thop**********************************")
