import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

class Pyramid_Scene_Transformer(nn.Module):
    def __init__(self, in_channels, embedding_dim=160, num_heads=4, norm='linear', num_layers=4, sizes=[[8,10], [4,5], [2,3]], args=None):
        super(Pyramid_Scene_Transformer, self).__init__()

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.sizes = sizes

        # 定义每个patch size对应的PST_Path模块 
        self.stage_1 = PST_Path(in_channels, [8, 10], embedding_dim, num_heads, num_layers, args)
        self.stage_2 = PST_Path(in_channels, [4, 5], embedding_dim, num_heads, num_layers, args)
        self.stage_3 = PST_Path(in_channels, [2, 3], embedding_dim, num_heads, num_layers, args)

        # Bottleneck层: 将所有尺度的特征拼接后恢复到原始通道数
        self.bottleneck = nn.Sequential(
            nn.Conv2d(embedding_dim * len(sizes), in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU()
        )

        # Regressor: 用于最后的预测
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
        )

    def forward(self, x):
        # 获取输入图像的批次大小和尺寸
        batch_size, channels, height, width = x.shape

        # 通过每个patch大小对应的PST_Path提取特征
        output_1 = self.stage_1(x)  # 第一尺度 [8, 10]  # torch.Size([81, 1, 160])
        output_2 = self.stage_2(x)  # 第二尺度 [4, 5]   # torch.Size([21, 1, 160])
        output_3 = self.stage_3(x)  # 第三尺度 [2, 3]   # torch.Size([7, 1, 160])

        # 提取第一个尺度的输出的第一个特征（通常是特殊token）
        first_token = output_1[0, ...]  # 提取第一个特征 # torch.Size([1, 160])

        # 对每个尺度的输出进行处理
        processed_outputs = []

        # 处理第一个尺度的输出
        output_1 = output_1[1:, ...]  # 跳过第一个token  # torch.Size([80, 1, 160])
        output_1 = output_1.permute(1, 2, 0).reshape(batch_size, self.embedding_dim, *self.sizes[0])  # 重新排列维度 torch.Size([1, 160, 8, 10])
        output_1 = F.interpolate(output_1, size=(height, width), mode='nearest')  # 上采样到原始图像大小 torch.Size([1, 160, 8, 10])
        processed_outputs.append(output_1)

        # 处理第二个尺度的输出
        output_2 = output_2[1:, ...]  # 跳过第一个token #torch.Size([20, 1, 160])
        output_2 = output_2.permute(1, 2, 0).reshape(batch_size, self.embedding_dim, *self.sizes[1])  # 重新排列维度 torch.Size([1, 160, 4, 5])
        output_2 = F.interpolate(output_2, size=(height, width), mode='nearest')  # 上采样到原始图像大小 # torch.Size([1, 160, 8, 10])
        processed_outputs.append(output_2)

        # 处理第三个尺度的输出
        output_3 = output_3[1:, ...]  # 跳过第一个token # torch.Size([6, 1, 160])
        output_3 = output_3.permute(1, 2, 0).reshape(batch_size, self.embedding_dim, *self.sizes[2])  # 重新排列维度 torch.Size([1, 160, 2, 3])
        output_3 = F.interpolate(output_3, size=(height, width), mode='nearest')  # 上采样到原始图像大小 torch.Size([1, 160, 8, 10])
        processed_outputs.append(output_3)

        # 将所有尺度的特征拼接在一起
        combined_features = torch.cat(processed_outputs, dim=1)

        # 通过bottleneck层处理拼接后的特征
        bottleneck_output = self.bottleneck(combined_features)

        # 通过regressor处理第一个尺度的token特征
        regressor_output = self.regressor(first_token)

        # 归一化处理
        eps = 0.1
        regressor_output = regressor_output + eps
        regressor_output = regressor_output / regressor_output.sum(dim=1, keepdim=True)

        return regressor_output, bottleneck_output



# import torch
# import torch.nn as nn
# from .layers import *
# import torch.nn.functional as F
# class Pyramid_Scene_Transformer(nn.Module):
#     def __init__(self, in_channels, embedding_dim=160,num_heads=4, norm='linear',num_layers=4,sizes=[[8,10],[4,5],[2,3]],args=None):
#         super(Pyramid_Scene_Transformer, self).__init__()
#         self.norm = norm
#         self.in_channels=in_channels
#         self.sizes=sizes
#         self.embedding_dim = embedding_dim
#         self.paths=nn.ModuleList([self._make_stage(in_channels, self.embedding_dim, size, num_heads, num_layers,args) for size in self.sizes])
#         self.bottleneck=nn.Sequential(
#             nn.Conv2d(self.embedding_dim*len(self.sizes), in_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.LeakyReLU()
#         )
#         self.regressor=nn.Sequential(
#             nn.Linear(self.embedding_dim, 128),
#             nn.LeakyReLU(),
#             nn.Linear(128, 128),
#             nn.LeakyReLU(),
#             nn.Linear(128, 256),
#         )

#     def _make_stage(self, in_channels,embedding_dim, patch_size, num_heads,num_layers,args):
#         tranformer=PST_Path(in_channels, patch_size, embedding_dim, num_heads,num_layers,args)
#         return nn.Sequential(tranformer)

#     def forward(self, x):
#         b, c, h, w=x.shape
#         priors=[path(x) for path in self.paths] # path 是一个 nn.Sequential，包含 PST_Path 调用 path(x) 会依次执行 nn.Sequential 中的所有模块
#         bin=priors[0][0,...] # 序列的第一个特征（通常是特殊token）?
#         # Change from S, N, E to N, S, E
#         for i in range(len(priors)):
#             priors[i]=priors[i][1:,...].permute(1, 2, 0).reshape(b,self.embedding_dim,*self.sizes[i])
#             #priors[i]=F.interpolate(priors[i],size=(h,w),mode='bilinear',align_corners=True)
#             priors[i]=F.interpolate(priors[i], size=(h, w), mode='nearest')

#         bottle = self.bottleneck(torch.cat(priors, 1))
#         y=self.regressor(bin)
#         # 将结果通过归一化处理
#         eps=0.1
#         y=y+eps
#         y=y/y.sum(dim=1, keepdim=True)

#         return y,bottle




