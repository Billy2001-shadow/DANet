# import torch
# from torchview import draw_graph
# from torchinfo import summary
# from models.net import DANet
# from options import get_args

# args = get_args('test')

# # 第一步：加载并运行一个模型
# x = torch.rand(1, 3, 228, 304)
# model = DANet(args)

# # 生成模型可视化图
# model_graph = draw_graph(model, input_size=x.shape, expand_nested=True, save_graph=True, filename="torchview", directory=".")
# model_graph.visual_graph

# # 使用 torchinfo 打印每个模块的参数量以及总的参数量
# summary(model, input_size=(1, 3, 228, 304))


import torch
from thop import profile
from fvcore.nn import FlopCountAnalysis
from models.net import DANet
from options import get_args

args = get_args('test')
model = DANet(args)
x = torch.rand(1, 3, 228, 304)  # 输入图像尺寸
flops, params = profile(model, inputs=(x,))
print("******************************thop**********************************")
print(f"FLOPs: {flops}")  # 输出 FLOPs
print(f"Parameters: {params}")  # 输出参数量
print("******************************thop**********************************")
