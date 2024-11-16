import torch
import torch.nn as nn
import timm

class MobileNetV4(nn.Module):
    def __init__(self):
        super(MobileNetV4, self).__init__()

        # 加载MobileNetV4模型，不包括分类头
        print('Loading MobileNetV4 base model...')
        # basemodel_name = 'mobilenetv4_conv_small.e2400_r224_in1k'
        basemodel_name = 'mobilenetv4_conv_medium.e500_r256_in1k'
        # 加载预训练的MobileNetV4模型
        basemodel = timm.create_model(basemodel_name, pretrained=False, features_only=True)
        # 离线加载权重文件，忽略分类头部分
        # pretrained_weights = torch.load('/home/chenwu/DANet/pytorch_model.bin')
        pretrained_weights = torch.load('/home/chenwu/DANet/mobilenetv4_conv_medium.e500_r256_in1k.bin')

        # 移除 head 层的键
        for key in list(pretrained_weights.keys()):
            if "head" in key or "classifier" in key:
                del pretrained_weights[key]

        basemodel.load_state_dict(pretrained_weights, strict=False)
        print('Done.')

        # 提取五个特征层
        # self.encoder = Encoder(basemodel)
        self.encoder = basemodel

    def forward(self, x):
        features = self.encoder(x)
        return features  

# class Encoder(nn.Module):
#     def __init__(self, backend):
#         super(Encoder, self).__init__()
#         self.original_model = backend

#     def forward(self, x):
#         features = [x]
#         for k, v in self.original_model._modules.items():
#             if k == 'blocks':
#                 for ki, vi in v._modules.items():
#                     features.append(vi(features[-1]))
#             else:
#                 features.append(v(features[-1]))

#         # 提取MobileNetV4的五个特征层
#         # 这里假设我们需要提取的层是output[1], output[2], output[3], output[4], output[5]
#         # 注意：不同的模型和不同版本的timm可能会略有不同，因此你需要根据实际情况选择要提取的层
#         features = [features[1], features[2], features[3], features[4], features[5]]
        
#         return features