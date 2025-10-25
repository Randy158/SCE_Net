
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_factory import get_resnet_backbone
from functools import partial
from Mamba import VSSMEncoder
nonlinearity = partial(F.relu, inplace=True)
import math

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.g = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        g = self.g(x)
        theta = self.theta(x)
        phi = self.phi(x)

        attn_map = torch.matmul(theta.flatten(2).transpose(1, 2), phi.flatten(2))
        attn_map = F.softmax(attn_map, dim=-1)
        attn_map = attn_map.transpose(1, 2)

        out = torch.matmul(attn_map, g.flatten(2))
        out = out.view_as(x)
class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class LightweightSPDACblock(nn.Module):
    def __init__(self, in_channels):
        super(LightweightSPDACblock, self).__init__()
        
        # 1. Depthwise Separable Convolutions: reduce parameter count and computation.
        self.depthwise_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        self.depthwise_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels, bias=False)
        self.pointwise_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        self.depthwise_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3, groups=in_channels, bias=False)
        self.pointwise_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # 2. Simplified Pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        
        # 3. Output convolution to fuse the features
        #self.conv_output = nn.Conv2d(in_channels * 6, in_channels, kernel_size=1)
        self.conv_output = nn.Conv2d(3072, in_channels, kernel_size=1)

    def forward(self, x):
        # Depthwise convolutions for feature extraction
        x1 = F.relu(self.pointwise_conv1(self.depthwise_conv1(x)))
        x2 = F.relu(self.pointwise_conv2(self.depthwise_conv2(x)))
        x3 = F.relu(self.pointwise_conv3(self.depthwise_conv3(x)))

        h, w = x.size(2), x.size(3)
        pooled1 = F.interpolate(self.pool1(x), size=(h, w), mode='bilinear', align_corners=True)  # Fix size mismatch
        pooled2 = F.interpolate(self.pool2(x), size=(h, w), mode='bilinear', align_corners=True)  # Fix size mismatch

        # Concatenate all the features (original + depthwise convolutions + pooling)
        out = torch.cat([x, x1, x2, x3, pooled1, pooled2], 1)

        # Final convolution to fuse and reduce the channels
        out = self.conv_output(out)
        return out

class DL_Net(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DL_Net, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.spda = LightweightSPDACblock(512)
        self.decoder4 = DecoderBlock(512, filters[2]) #这是ALL版本
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.conv = nn.Conv2d(1, 3, kernel_size=1, stride=1)
        self.vssm_encoder = VSSMEncoder(patch_size=2, in_chans=48)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(48, eps=1e-5, affine=True),
        )
        
    def forward(self, x):
        ssmx = self.stem(x)
        b, c, h, w = x.shape
        if c == 1:
            x = self.conv(x)
        else:
            pass
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
       
        vssm_feats = self.vssm_encoder(ssmx)  
        v1, v2, v3, v4 = vssm_feats[1:5]  

        e1 = e1 + v1
        e2 = e2 + v2
        e3 = e3 + v3
        e4 = e4 + v4
        e4 = self.spda(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return torch.sigmoid(out)


if __name__ == "__main__":
    x = torch.randn(8, 3, 256, 256)
    model = DL_Net(num_classes=2)
    result = model(x)
    print(result.shape)
