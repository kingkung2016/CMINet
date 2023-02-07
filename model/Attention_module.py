
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CRM, self).__init__()

        self.conv_RGB = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1), nn.BatchNorm2d(out_channel), nn.ReLU(),
        )

        self.conv_depth = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1), nn.BatchNorm2d(out_channel), nn.ReLU(),
        )

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, 1, 1), nn.BatchNorm2d(out_channel), nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 1, 1, 0), nn.BatchNorm2d(out_channel), nn.ReLU()
        )

        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.channel_reduce = nn.Sequential(
            nn.Conv2d(out_channel*3, out_channel, 3, 1, 1), nn.BatchNorm2d(out_channel), nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 1, 1, 0), nn.BatchNorm2d(out_channel), nn.ReLU()
        )


    def forward(self, RGB, depth):
        RGB = self.conv_RGB(RGB)
        depth = self.conv_depth(depth)

        RGBD = torch.cat((RGB, depth), dim=1)
        RGBD = self.conv_fusion(RGBD)

        RGB_CA = self.channel_attention_rgb(self.squeeze_rgb(RGB))
        RGB_weight = RGB * RGB_CA.expand_as(RGB)

        Depth_CA = self.channel_attention_depth(self.squeeze_depth(depth))
        Depth_weight = depth * Depth_CA.expand_as(depth)

        Fusion_CA = torch.softmax(RGB_CA + Depth_CA, dim=1)
        RGBD_weight = RGBD * Fusion_CA.expand_as(RGBD)

        output = torch.cat((RGB_weight, Depth_weight, RGBD_weight), dim=1)
        output = self.channel_reduce(output)

        return output


class StripPooling(nn.Module):
    def __init__(self, in_channels):
        super(StripPooling, self).__init__()

        self.pool_row = nn.AdaptiveAvgPool2d((1, None))
        self.pool_column = nn.AdaptiveAvgPool2d((None, 1))

        self.conv_row = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1, 3), 1, (0, 1), bias=False), nn.BatchNorm2d(in_channels))
        self.conv_column = nn.Sequential(nn.Conv2d(in_channels, in_channels, (3, 1), 1, (1, 0), bias=False), nn.BatchNorm2d(in_channels))
        self.conv_weight = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        x_row = F.interpolate(self.conv_row(self.pool_row(x)), (h, w), mode='bilinear', align_corners=True)
        x_column = F.interpolate(self.conv_column(self.pool_column(x)), (h, w), mode='bilinear', align_corners=True)

        #先用卷积再用池化
        weight = self.sigmoid(self.conv_weight(x_row + x_column))
        out = weight * x
        #考虑使用残差连接

        return out


class Mixed_Pooling(nn.Module):
    def __init__(self, in_channel, pool_size):
        super(Mixed_Pooling, self).__init__()
        self.pool_size = pool_size
        self.image_size = 256

        #第一个stage
        self.strip_1 = StripPooling(in_channel[0])
        self.pool1_2 = nn.AdaptiveAvgPool2d(self.pool_size[1])
        self.pool1_3 = nn.AdaptiveAvgPool2d(self.pool_size[2])
        self.pool1_4 = nn.AdaptiveAvgPool2d(self.pool_size[3])

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channel[0], in_channel[0], 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel[0]),
                                     nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channel[0], in_channel[0], 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel[0]),
                                     nn.ReLU(True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channel[0], in_channel[0], 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel[0]),
                                     nn.ReLU(True))
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channel[0], in_channel[0], 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel[0]),
                                     nn.ReLU(True))

        #第二个stage
        self.strip_2 = StripPooling(in_channel[1])
        self.pool2_3 = nn.AdaptiveAvgPool2d(self.pool_size[2])
        self.pool2_4 = nn.AdaptiveAvgPool2d(self.pool_size[3])

        self.conv2_2 = nn.Sequential(nn.Conv2d(in_channel[1], in_channel[0], 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel[0]),
                                     nn.ReLU(True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(in_channel[1], in_channel[0], 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel[0]),
                                     nn.ReLU(True))
        self.conv2_4 = nn.Sequential(nn.Conv2d(in_channel[1], in_channel[0], 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel[0]),
                                     nn.ReLU(True))

        #第三个stage
        self.strip_3 = StripPooling(in_channel[2])
        self.pool3_4 = nn.AdaptiveAvgPool2d(self.pool_size[3])

        self.conv3_3 = nn.Sequential(nn.Conv2d(in_channel[2], in_channel[1], 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel[1]),
                                     nn.ReLU(True))
        self.conv3_4 = nn.Sequential(nn.Conv2d(in_channel[2], in_channel[1], 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel[1]),
                                     nn.ReLU(True))

        #第四个stage
        self.strip_4 = StripPooling(in_channel[3])
        self.conv4_4 = nn.Sequential(nn.Conv2d(in_channel[3], in_channel[2], 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel[2]),
                                     nn.ReLU(True))

        self.conv_out_4 = nn.Sequential(nn.Conv2d(in_channel[3], in_channel[0], 1, 1, 0), nn.BatchNorm2d(in_channel[0]), nn.ReLU(True),
                                        nn.Conv2d(in_channel[0], 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(True))

        self.conv_out_3 = nn.Sequential(nn.Conv2d(in_channel[2], in_channel[0], 1, 1, 0), nn.BatchNorm2d(in_channel[0]), nn.ReLU(True),
                                        nn.Conv2d(in_channel[0], 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(True))

        self.conv_out_2 = nn.Sequential(nn.Conv2d(in_channel[1], in_channel[0], 1, 1, 0), nn.BatchNorm2d(in_channel[0]), nn.ReLU(True),
                                        nn.Conv2d(in_channel[0], 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(True))

        self.conv_out_1 = nn.Sequential(nn.Conv2d(in_channel[0], in_channel[0], 1, 1, 0), nn.BatchNorm2d(in_channel[0]), nn.ReLU(True),
                                        nn.Conv2d(in_channel[0], 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(True))

    def forward(self, F1, F2, F3, F4):
        _, _, H1, W1 = F1.size()
        F1_1 = self.conv1_1(self.strip_1(F1))
        F1_2 = self.conv1_2(self.pool1_2(F1))
        F1_3 = self.conv1_3(self.pool1_3(F1))
        F1_4 = self.conv1_4(self.pool1_4(F1))

        F2_2 = self.conv2_2(self.strip_2(F2))
        F2_3 = self.conv2_3(self.pool2_3(F2))
        F2_4 = self.conv2_4(self.pool2_4(F2))

        F3_3 = self.conv3_3(self.strip_3(F3))
        F3_4 = self.conv3_4(self.pool3_4(F3))

        F4_4 = self.conv4_4(self.strip_4(F4))

        out_1 = self.conv_out_1(F1_1)
        out_2 = self.conv_out_2(torch.cat((F1_2, F2_2), dim=1))
        out_3 = self.conv_out_3(torch.cat((F1_3, F2_3, F3_3), dim=1))
        out_4 = self.conv_out_4(torch.cat((F1_4, F2_4, F3_4, F4_4), dim=1))
        #统一降低通道数，使用加法相加，softmax激活

        return out_1, out_2, out_3, out_4


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        self.asppconv = torch.nn.Sequential()
        if bn_start:
            self.asppconv = nn.Sequential(
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate),
                nn.BatchNorm2d(num2),
                nn.ReLU(inplace=True)
            )
        else:
            self.asppconv = nn.Sequential(
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate),
                nn.ReLU(inplace=True)
            )
        self.drop_rate = drop_out

    def forward(self, _input):
        #feature = super(_DenseAsppBlock, self).forward(_input)
        feature = self.asppconv(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class Multi_Dilated_Decoder(nn.Module):
    def __init__(self, channel, image_size = 256):
        super(Multi_Dilated_Decoder, self).__init__()
        self.image_size = image_size

        #
        self.dilated_conv_4_4 = _DenseAsppBlock(input_num=channel, num1=channel, num2=channel, dilation_rate=1,
                                                drop_out=0, bn_start=True)
        self.dilated_conv_4_3 = _DenseAsppBlock(input_num=channel, num1=channel, num2=channel, dilation_rate=3,
                                                drop_out=0, bn_start=True)
        self.dilated_conv_4_2 = _DenseAsppBlock(input_num=channel, num1=channel, num2=channel, dilation_rate=6,
                                                drop_out=0, bn_start=True)
        self.dilated_conv_4_1 = _DenseAsppBlock(input_num=channel, num1=channel, num2=channel, dilation_rate=12,
                                                drop_out=0, bn_start=True)

        #
        self.dilated_conv_3_3 = _DenseAsppBlock(input_num=channel*2, num1=channel*2, num2=channel, dilation_rate=1,
                                                drop_out=0, bn_start=True)
        self.dilated_conv_3_2 = _DenseAsppBlock(input_num=channel*2, num1=channel*2, num2=channel, dilation_rate=3,
                                                drop_out=0, bn_start=True)
        self.dilated_conv_3_1 = _DenseAsppBlock(input_num=channel*2, num1=channel*2, num2=channel, dilation_rate=6,
                                                drop_out=0, bn_start=True)

        #
        self.dilated_conv_2_2 = _DenseAsppBlock(input_num=channel*3, num1=channel*3, num2=channel, dilation_rate=1,
                                                drop_out=0, bn_start=True)
        self.dilated_conv_2_1 = _DenseAsppBlock(input_num=channel*3, num1=channel*3, num2=channel, dilation_rate=3,
                                                drop_out=0, bn_start=True)

        #
        self.dilated_conv_1_1 = _DenseAsppBlock(input_num=channel*4, num1=channel*4, num2=channel, dilation_rate=1,
                                                drop_out=0, bn_start=True)


        self.Sal_Head_1 = Sal_Head(channel)
        self.Sal_Head_2 = Sal_Head(channel)
        self.Sal_Head_3 = Sal_Head(channel)
        self.Sal_Head_4 = Sal_Head(channel)

    def forward(self, F1, F2, F3, F4):

        F4_out = self.dilated_conv_4_4(F4)
        F4_3 = self.dilated_conv_4_3(F.interpolate(F4, scale_factor=2, mode='bilinear', align_corners=False))
        F4_2 = self.dilated_conv_4_2(F.interpolate(F4, scale_factor=4, mode='bilinear', align_corners=False))
        F4_1 = self.dilated_conv_4_1(F.interpolate(F4, scale_factor=8, mode='bilinear', align_corners=False))

        F3 = torch.cat((F3, F4_3), dim=1)
        F3_out = self.dilated_conv_3_3(F3)
        F3_2 = self.dilated_conv_3_2(F.interpolate(F3, scale_factor=2, mode='bilinear', align_corners=False))
        F3_1 = self.dilated_conv_3_1(F.interpolate(F3, scale_factor=4, mode='bilinear', align_corners=False))

        F2 = torch.cat((F2, F4_2, F3_2), dim=1)
        F2_out = self.dilated_conv_2_2(F2)
        F2_1 = self.dilated_conv_2_1(F.interpolate(F2, scale_factor=2, mode='bilinear', align_corners=False))

        F1 = torch.cat((F1, F4_1, F3_1, F2_1), dim=1)
        F1_out = self.dilated_conv_1_1(F1)

        F4_out = F.interpolate(self.Sal_Head_4(F4_out), self.image_size, mode='bilinear', align_corners=False)
        F3_out = F.interpolate(self.Sal_Head_3(F3_out), self.image_size, mode='bilinear', align_corners=False)
        F2_out = F.interpolate(self.Sal_Head_2(F2_out), self.image_size, mode='bilinear', align_corners=False)
        F1_out = F.interpolate(self.Sal_Head_1(F1_out), self.image_size, mode='bilinear', align_corners=False)

        return F1_out, F2_out, F3_out, F4_out


class Sal_Head(nn.Module):
    def __init__(self, channel):
        super(Sal_Head, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, 1, 0)
        )

    def forward(self, x):
        y = self.conv(x)
        return y


if __name__== '__main__':

    F1, F2, F3, F4 = torch.randn(1,64,64,64), torch.randn(1,128,32,32), torch.randn(1,256,16,16), torch.randn(1,512,8,8)
    channel = [64, 128, 256, 512]
    pool_size = [64, 32, 16, 8]
    module = Mixed_Pooling(channel, pool_size)
    out_1, out_2, out_3, out_4 = module(F1, F2, F3, F4)
    print(out_1.shape, out_2.shape, out_3.shape, out_4.shape)

    # from DNN_printer import DNN_printer
    # DNN_printer(module, input_size=[(64,64,64), (128,32,32), (256,16,16), (512,8,8)], batch_size=1)


