
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import resnet
from model import Attention_module


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()

        self.RGB_net = resnet.resnet50(include_top=False)
        self.Depth_net = resnet.resnet50(include_top=False)

        #特征融合和选择
        self.channel_attention_1 = Attention_module.CRM(256, 128)
        self.channel_attention_2 = Attention_module.CRM(512, 256)
        self.channel_attention_3 = Attention_module.CRM(1024, 512)
        self.channel_attention_4 = Attention_module.CRM(2048, 1024)

        self.pool_size = [64, 32, 16, 8]
        self.Mixed_Spatial_attention = Attention_module.Mixed_Pooling([128, 256, 512, 1024], self.pool_size)

        self.decoder = Attention_module.Multi_Dilated_Decoder(64)

    def load_pretrained_model(self, RGB_model_path, Depth_model_path):
        RGB_pretrained_dict = torch.load(RGB_model_path)
        RGB_model_dict = self.RGB_net.state_dict()
        RGB_pretrained_dict = {k: v for k, v in RGB_pretrained_dict.items() if k in RGB_model_dict}
        RGB_model_dict.update(RGB_pretrained_dict)
        self.RGB_net.load_state_dict(RGB_pretrained_dict)

        #加载权重参数
        Depth_pretrained_dict = torch.load(Depth_model_path)
        #获取当前模型参数
        Depth_model_dict = self.Depth_net.state_dict()
        #将pretained_dict里不属于model_dict的键剔除掉
        Depth_pretrained_dict = {k: v for k, v in Depth_pretrained_dict.items() if k in Depth_model_dict}
        #更新模型权重
        Depth_model_dict.update(Depth_pretrained_dict)
        self.Depth_net.load_state_dict(Depth_pretrained_dict)


    def forward(self, RGB, depth):
        depth = depth.repeat(1, 3, 1, 1)  #把深度图变成3个通道
        image_size = RGB.size()[2:]

        #backbone提取特征
        Fr1, Fr2, Fr3, Fr4 = self.RGB_net(RGB)
        Fd1, Fd2, Fd3, Fd4 = self.Depth_net(depth)

        # F = 64*64*256, 32*32*512, 16*16*1024, 8*8*2048
        #print(Fu1.shape, Fu2.shape, Fu3.shape, Fu4.shape)

        #通道数融合注意力机制
        fusion_1 = self.channel_attention_1(Fr1, Fd1)
        fusion_2 = self.channel_attention_2(Fr2, Fd2)
        fusion_3 = self.channel_attention_3(Fr3, Fd3)
        fusion_4 = self.channel_attention_4(Fr4, Fd4)

        #多尺度混合空间池化
        F1, F2, F3, F4 = self.Mixed_Spatial_attention(fusion_1, fusion_2, fusion_3, fusion_4)

        #多尺度空洞卷积解码器
        out_1, out_2, out_3, out_4 = self.decoder(F1, F2, F3, F4)


        return out_1, out_2, out_3, out_4



if __name__== '__main__':
    rgb = torch.rand(1,3,256,256).cuda()
    depth = torch.rand(1,1,256,256).cuda()

    model_fusion = FusionNet().cuda()

    #使用ResNet模型
    ResNet_50_pretrained = '../pretrained/resnet_50.pth'
    model_fusion.load_pretrained_model(ResNet_50_pretrained, ResNet_50_pretrained)

    Fr1, Fr2, Fr3, Fr4 = model_fusion(rgb, depth)
    print(Fr1.shape, Fr2.shape, Fr3.shape, Fr4.shape)
