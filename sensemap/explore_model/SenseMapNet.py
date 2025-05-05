import torch
import torch.nn as nn
import torch.nn.functional as F
from explore_model.ffc import Bottleneck, FFC_BN_ACT

class FFCBlock(nn.Module):
    def __init__(self, in_channels):
        super(FFCBlock, self).__init__()
        self.groups = 1
        self.dilation = 1
        self.use_se = False
        self.lfu = True
        self.base_width = 64
        self.inplanes = in_channels
        self.block = Bottleneck
        self.block.expansion = 1

        self.layer1 = self._make_layer(self.inplanes*1, 2, stride=1, ratio_gin=0, ratio_gout=1)
        self.layer2 = self._make_layer(self.inplanes*1, 2, stride=1, ratio_gin=1, ratio_gout=0)

    def _make_layer(self, planes, blocks, stride=1, ratio_gin=0.5, ratio_gout=0.5):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.block.expansion or ratio_gin == 0:
            downsample = FFC_BN_ACT(self.inplanes, planes * self.block.expansion, kernel_size=1, stride=stride,
                                    ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=self.lfu)

        layers = []
        layers.append(self.block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                            self.dilation, ratio_gin, ratio_gout, lfu=self.lfu, use_se=self.use_se))
        self.inplanes = planes * self.block.expansion
        for _ in range(1, blocks):
            layers.append(self.block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout, lfu=self.lfu, use_se=self.use_se))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x[0]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5)
        )

    def forward(self, x):
        return self.conv(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        return x

    
class BilinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BilinearBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.dropout(x)
        return x


class DistillMapNet(nn.Module):
    def __init__(self, image_size=256, dim=4):
        super(DistillMapNet, self).__init__()

        self.image_size = image_size
        self.dim = dim

        # 编码器
        self.conv_encoder1 = nn.Sequential(SeparableConv2d(3, self.dim), SeparableConv2d(self.dim, self.dim), SeparableConv2d(self.dim, self.dim))
        self.conv_encoder2 = nn.Sequential(SeparableConv2d(self.dim, 2*self.dim, stride=2), FFCBlock(2*self.dim), SeparableConv2d(2*self.dim, 2*self.dim))
        self.conv_encoder3 = nn.Sequential(SeparableConv2d(2*self.dim, 4*self.dim, stride=2), FFCBlock(4*self.dim), SeparableConv2d(4*self.dim, 4*self.dim))
        self.conv_encoder4 = nn.Sequential(SeparableConv2d(4*self.dim, 8*self.dim, stride=2), FFCBlock(8*self.dim), SeparableConv2d(8*self.dim, 8*self.dim))
        self.conv_encoder5 = nn.Sequential(SeparableConv2d(8*self.dim, 16*self.dim, stride=2), FFCBlock(16*self.dim), SeparableConv2d(16*self.dim, 16*self.dim))

        # 解码器
        self.upsample1 = BilinearBlock(16*self.dim, 8*self.dim)
        self.decoder1 = nn.Sequential(SeparableConv2d(16*self.dim, 8*self.dim), SeparableConv2d(8*self.dim, 8*self.dim))
        self.upsample2 = BilinearBlock(8*self.dim, 4*self.dim)
        self.decoder2 = nn.Sequential(SeparableConv2d(8*self.dim, 4*self.dim), SeparableConv2d(4*self.dim, 4*self.dim))
        self.upsample3 = BilinearBlock(4*self.dim, 2*self.dim)
        self.decoder3 = nn.Sequential(SeparableConv2d(4*self.dim, 2*self.dim), SeparableConv2d(2*self.dim, 2*self.dim))
        self.upsample4 = BilinearBlock(2*self.dim, self.dim)
        self.decoder4 = nn.Sequential(SeparableConv2d(2*self.dim, self.dim), SeparableConv2d(self.dim, self.dim), nn.Conv2d(self.dim, 1, kernel_size=1, bias=False))

    def forward(self, x):

        # 编码器
        conv1 = self.conv_encoder1(x)
        conv2 = self.conv_encoder2(conv1)
        conv3 = self.conv_encoder3(conv2)
        conv4 = self.conv_encoder4(conv3)
        conv5 = self.conv_encoder5(conv4)

        # 解码器
        decode1 = self.upsample1(conv5)
        decode1 = self.decoder1(torch.cat([decode1, conv4], dim=1))
        decode2 = self.upsample2(decode1)
        decode2 = self.decoder2(torch.cat([decode2, conv3], dim=1))
        decode3 = self.upsample3(decode2)
        decode3 = self.decoder3(torch.cat([decode3, conv2], dim=1))
        decode4 = self.upsample4(decode3)
        decode4 = self.decoder4(torch.cat([decode4, conv1], dim=1))
        
        out = F.sigmoid(decode4)

        return out, conv2, conv3, decode2, decode3
    
class DistillMapNetDeconv(nn.Module):
    def __init__(self, image_size=256, dim=4):
        super(DistillMapNetDeconv, self).__init__()

        self.image_size = image_size
        self.dim = dim

        # 编码器
        self.conv_encoder1 = nn.Sequential(SeparableConv2d(3, self.dim), SeparableConv2d(self.dim, self.dim), SeparableConv2d(self.dim, self.dim))
        self.conv_encoder2 = nn.Sequential(SeparableConv2d(self.dim, 2*self.dim, stride=2), FFCBlock(2*self.dim), SeparableConv2d(2*self.dim, 2*self.dim))
        self.conv_encoder3 = nn.Sequential(SeparableConv2d(2*self.dim, 4*self.dim, stride=2), FFCBlock(4*self.dim), SeparableConv2d(4*self.dim, 4*self.dim))
        self.conv_encoder4 = nn.Sequential(SeparableConv2d(4*self.dim, 8*self.dim, stride=2), FFCBlock(8*self.dim), SeparableConv2d(8*self.dim, 8*self.dim))
        self.conv_encoder5 = nn.Sequential(SeparableConv2d(8*self.dim, 16*self.dim, stride=2), FFCBlock(16*self.dim), SeparableConv2d(16*self.dim, 16*self.dim))

        # 解码器
        self.upsample1 = DeconvBlock(16*self.dim, 8*self.dim)
        self.decoder1 = nn.Sequential(SeparableConv2d(16*self.dim, 8*self.dim), SeparableConv2d(8*self.dim, 8*self.dim))
        self.upsample2 = DeconvBlock(8*self.dim, 4*self.dim)
        self.decoder2 = nn.Sequential(SeparableConv2d(8*self.dim, 4*self.dim), SeparableConv2d(4*self.dim, 4*self.dim))
        self.upsample3 = DeconvBlock(4*self.dim, 2*self.dim)
        self.decoder3 = nn.Sequential(SeparableConv2d(4*self.dim, 2*self.dim), SeparableConv2d(2*self.dim, 2*self.dim))
        self.upsample4 = DeconvBlock(2*self.dim, self.dim)
        self.decoder4 = nn.Sequential(SeparableConv2d(2*self.dim, self.dim), SeparableConv2d(self.dim, self.dim), nn.Conv2d(self.dim, 1, kernel_size=1, bias=False))

    def forward(self, x):

        # 编码器
        conv1 = self.conv_encoder1(x)
        conv2 = self.conv_encoder2(conv1)
        conv3 = self.conv_encoder3(conv2)
        conv4 = self.conv_encoder4(conv3)
        conv5 = self.conv_encoder5(conv4)

        # 解码器
        decode1 = self.upsample1(conv5)
        decode1 = self.decoder1(torch.cat([decode1, conv4], dim=1))
        decode2 = self.upsample2(decode1)
        decode2 = self.decoder2(torch.cat([decode2, conv3], dim=1))
        decode3 = self.upsample3(decode2)
        decode3 = self.decoder3(torch.cat([decode3, conv2], dim=1))
        decode4 = self.upsample4(decode3)
        decode4 = self.decoder4(torch.cat([decode4, conv1], dim=1))
        
        out = F.sigmoid(decode4)

        return out, conv2, conv3, decode2, decode3
    
class DistillMapNetNormal(nn.Module):
    def __init__(self, image_size=256, dim=4):
        super(DistillMapNetNormal, self).__init__()

        self.image_size = image_size
        self.dim = dim

        # 编码器
        self.conv_encoder1 = nn.Sequential(ConvBlock(3, self.dim), ConvBlock(self.dim, self.dim), ConvBlock(self.dim, self.dim))
        self.conv_encoder2 = nn.Sequential(ConvBlock(self.dim, 2*self.dim, stride=2), FFCBlock(2*self.dim), ConvBlock(2*self.dim, 2*self.dim))
        self.conv_encoder3 = nn.Sequential(ConvBlock(2*self.dim, 4*self.dim, stride=2), FFCBlock(4*self.dim), ConvBlock(4*self.dim, 4*self.dim))
        self.conv_encoder4 = nn.Sequential(ConvBlock(4*self.dim, 8*self.dim, stride=2), FFCBlock(8*self.dim), ConvBlock(8*self.dim, 8*self.dim))
        self.conv_encoder5 = nn.Sequential(ConvBlock(8*self.dim, 16*self.dim, stride=2), FFCBlock(16*self.dim), ConvBlock(16*self.dim, 16*self.dim))

        # 解码器
        self.upsample1 = BilinearBlock(16*self.dim, 8*self.dim)
        self.decoder1 = nn.Sequential(ConvBlock(16*self.dim, 8*self.dim), ConvBlock(8*self.dim, 8*self.dim))
        self.upsample2 = BilinearBlock(8*self.dim, 4*self.dim)
        self.decoder2 = nn.Sequential(ConvBlock(8*self.dim, 4*self.dim), ConvBlock(4*self.dim, 4*self.dim))
        self.upsample3 = BilinearBlock(4*self.dim, 2*self.dim)
        self.decoder3 = nn.Sequential(ConvBlock(4*self.dim, 2*self.dim), ConvBlock(2*self.dim, 2*self.dim))
        self.upsample4 = BilinearBlock(2*self.dim, self.dim)
        self.decoder4 = nn.Sequential(ConvBlock(2*self.dim, self.dim), ConvBlock(self.dim, self.dim), nn.Conv2d(self.dim, 1, kernel_size=1, bias=False))

    def forward(self, x):

        # 编码器
        conv1 = self.conv_encoder1(x)
        conv2 = self.conv_encoder2(conv1)
        conv3 = self.conv_encoder3(conv2)
        conv4 = self.conv_encoder4(conv3)
        conv5 = self.conv_encoder5(conv4)

        # 解码器
        decode1 = self.upsample1(conv5)
        decode1 = self.decoder1(torch.cat([decode1, conv4], dim=1))
        decode2 = self.upsample2(decode1)
        decode2 = self.decoder2(torch.cat([decode2, conv3], dim=1))
        decode3 = self.upsample3(decode2)
        decode3 = self.decoder3(torch.cat([decode3, conv2], dim=1))
        decode4 = self.upsample4(decode3)
        decode4 = self.decoder4(torch.cat([decode4, conv1], dim=1))
        
        out = F.sigmoid(decode4)

        return out, conv2, conv3, decode2, decode3


class TeacherMapNet(nn.Module):
    def __init__(self, image_size=256, dim=4):
        super(TeacherMapNet, self).__init__()

        self.image_size = image_size
        self.dim = dim

        # 编码器
        self.conv_encoder1 = nn.Sequential(ConvBlock(3, self.dim), ConvBlock(self.dim, self.dim), ConvBlock(self.dim, self.dim))
        self.conv_encoder2 = nn.Sequential(ConvBlock(self.dim, 2*self.dim, stride=2), FFCBlock(2*self.dim), ConvBlock(2*self.dim, 2*self.dim), FFCBlock(2*self.dim), ConvBlock(2*self.dim, 2*self.dim))
        self.conv_encoder3 = nn.Sequential(ConvBlock(2*self.dim, 4*self.dim, stride=2), FFCBlock(4*self.dim), ConvBlock(4*self.dim, 4*self.dim), FFCBlock(4*self.dim), ConvBlock(4*self.dim, 4*self.dim))
        self.conv_encoder4 = nn.Sequential(ConvBlock(4*self.dim, 8*self.dim, stride=2), FFCBlock(8*self.dim), ConvBlock(8*self.dim, 8*self.dim), FFCBlock(8*self.dim), ConvBlock(8*self.dim, 8*self.dim))
        self.conv_encoder5 = nn.Sequential(ConvBlock(8*self.dim, 16*self.dim, stride=2), FFCBlock(16*self.dim), ConvBlock(16*self.dim, 16*self.dim), FFCBlock(16*self.dim), ConvBlock(16*self.dim, 16*self.dim))

        # 解码器
        self.upsample1 = BilinearBlock(16*self.dim, 8*self.dim)
        self.decoder1 = nn.Sequential(ConvBlock(16*self.dim, 8*self.dim), ConvBlock(8*self.dim, 8*self.dim), ConvBlock(8*self.dim, 8*self.dim), ConvBlock(8*self.dim, 8*self.dim))
        self.upsample2 = BilinearBlock(8*self.dim, 4*self.dim)
        self.decoder2 = nn.Sequential(ConvBlock(8*self.dim, 4*self.dim), ConvBlock(4*self.dim, 4*self.dim), ConvBlock(4*self.dim, 4*self.dim), ConvBlock(4*self.dim, 4*self.dim))
        self.upsample3 = BilinearBlock(4*self.dim, 2*self.dim)
        self.decoder3 = nn.Sequential(ConvBlock(4*self.dim, 2*self.dim), ConvBlock(2*self.dim, 2*self.dim), ConvBlock(2*self.dim, 2*self.dim), ConvBlock(2*self.dim, 2*self.dim))
        self.upsample4 = BilinearBlock(2*self.dim, self.dim)
        self.decoder4 = nn.Sequential(ConvBlock(2*self.dim, self.dim), ConvBlock(self.dim, self.dim), nn.Conv2d(self.dim, 1, kernel_size=1, bias=False))

    def forward(self, x):

        # 编码器
        conv1 = self.conv_encoder1(x)
        conv2 = self.conv_encoder2(conv1)
        conv3 = self.conv_encoder3(conv2)
        conv4 = self.conv_encoder4(conv3)
        conv5 = self.conv_encoder5(conv4)

        # 解码器
        decode1 = self.upsample1(conv5)
        decode1 = self.decoder1(torch.cat([decode1, conv4], dim=1))
        decode2 = self.upsample2(decode1)
        decode2 = self.decoder2(torch.cat([decode2, conv3], dim=1))
        decode3 = self.upsample3(decode2)
        decode3 = self.decoder3(torch.cat([decode3, conv2], dim=1))
        decode4 = self.upsample4(decode3)
        decode4 = self.decoder4(torch.cat([decode4, conv1], dim=1))
        
        out = F.sigmoid(decode4)

        return out, conv2, conv3, decode2, decode3
    
class TeacherMapNet2(nn.Module):
    def __init__(self, image_size=256, dim=4):
        super(TeacherMapNet2, self).__init__()

        self.image_size = image_size
        self.dim = dim

        # 编码器
        self.conv_encoder1 = nn.Sequential(ConvBlock(3, self.dim), ConvBlock(self.dim, self.dim), ConvBlock(self.dim, self.dim))
        self.conv_encoder2 = nn.Sequential(ConvBlock(self.dim, 2*self.dim, stride=2), FFCBlock(2*self.dim), ConvBlock(2*self.dim, 2*self.dim), FFCBlock(2*self.dim), ConvBlock(2*self.dim, 2*self.dim))
        self.conv_encoder3 = nn.Sequential(ConvBlock(2*self.dim, 4*self.dim, stride=2), FFCBlock(4*self.dim), ConvBlock(4*self.dim, 4*self.dim), FFCBlock(4*self.dim), ConvBlock(4*self.dim, 4*self.dim))
        self.conv_encoder4 = nn.Sequential(ConvBlock(4*self.dim, 8*self.dim, stride=2), FFCBlock(8*self.dim), ConvBlock(8*self.dim, 8*self.dim), FFCBlock(8*self.dim), ConvBlock(8*self.dim, 8*self.dim))
        self.conv_encoder5 = nn.Sequential(ConvBlock(8*self.dim, 16*self.dim, stride=2), FFCBlock(16*self.dim), ConvBlock(16*self.dim, 16*self.dim), FFCBlock(16*self.dim), ConvBlock(16*self.dim, 16*self.dim))

        # 解码器
        self.upsample1 = DeconvBlock(16*self.dim, 8*self.dim)
        self.decoder1 = nn.Sequential(ConvBlock(16*self.dim, 8*self.dim), ConvBlock(8*self.dim, 8*self.dim), ConvBlock(8*self.dim, 8*self.dim), ConvBlock(8*self.dim, 8*self.dim))
        self.upsample2 = DeconvBlock(8*self.dim, 4*self.dim)
        self.decoder2 = nn.Sequential(ConvBlock(8*self.dim, 4*self.dim), ConvBlock(4*self.dim, 4*self.dim), ConvBlock(4*self.dim, 4*self.dim), ConvBlock(4*self.dim, 4*self.dim))
        self.upsample3 = DeconvBlock(4*self.dim, 2*self.dim)
        self.decoder3 = nn.Sequential(ConvBlock(4*self.dim, 2*self.dim), ConvBlock(2*self.dim, 2*self.dim), ConvBlock(2*self.dim, 2*self.dim), ConvBlock(2*self.dim, 2*self.dim))
        self.upsample4 = DeconvBlock(2*self.dim, self.dim)
        self.decoder4 = nn.Sequential(ConvBlock(2*self.dim, self.dim), ConvBlock(self.dim, self.dim), nn.Conv2d(self.dim, 1, kernel_size=1, bias=False))

    def forward(self, x):

        # 编码器
        conv1 = self.conv_encoder1(x)
        conv2 = self.conv_encoder2(conv1)
        conv3 = self.conv_encoder3(conv2)
        conv4 = self.conv_encoder4(conv3)
        conv5 = self.conv_encoder5(conv4)

        # 解码器
        decode1 = self.upsample1(conv5)
        decode1 = self.decoder1(torch.cat([decode1, conv4], dim=1))
        decode2 = self.upsample2(decode1)
        decode2 = self.decoder2(torch.cat([decode2, conv3], dim=1))
        decode3 = self.upsample3(decode2)
        decode3 = self.decoder3(torch.cat([decode3, conv2], dim=1))
        decode4 = self.upsample4(decode3)
        decode4 = self.decoder4(torch.cat([decode4, conv1], dim=1))
        
        out = F.sigmoid(decode4)

        return out, conv2, conv3, decode2, decode3