import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from resnet.resnet_architecture import get_resnet_backbone
from functools import partial
from model.Mamba import VSSMEncoder
nonlinearity = partial(F.relu, inplace=True)
import math

class DynamicFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channel * 2, channel // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel // 2, 2, 3, padding=1),
            nn.Softmax(dim=1))

    def forward(self, resnet_feat, vssm_feat):
        gate = self.gate(torch.cat([resnet_feat, vssm_feat], dim=1))
        return gate[:, 0:1] * resnet_feat + gate[:, 1:2] * vssm_feat

class EdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        edge_map = self.sigmoid(self.conv(x))
        return x * edge_map

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4 + skip_channels, out_channels, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nonlinearity

    def forward(self, x, skip):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class SCE_Net(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
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
    
        self.decoder4 = DecoderBlock(512, 256, 256)  
        self.decoder3 = DecoderBlock(256, 128, 128)  
        self.decoder2 = DecoderBlock(128, 64, 64)  
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
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

        self.fusion1 = DynamicFusion(64)
        self.fusion2 = DynamicFusion(128)
        self.fusion3 = DynamicFusion(256)
        self.fusion4 = DynamicFusion(512)
 
        self.edge3 = EdgeAttention(512)
        self.edge2 = EdgeAttention(256)
   
        self.vss_pre_cnn = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.vss_post_cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(48, 48, kernel_size=3, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
        ])

    def forward(self, x):
        ssmx = self.stem(x)  # [B, 48, H/2, W/2]
        # Before VSS
        ssmx = self.vss_pre_cnn(ssmx)  # [16, 48, 128, 128]
        # VSSM Encoder
        vssm_feats = self.vssm_encoder(ssmx)  # [B, C, H, W]
        # After VSS
        vssm_feats = [self.vss_post_cnn[i](vssm_feats[i]) for i in range(5)]
        # v1~v4
        v0, v1, v2, v3, v4 = vssm_feats[0:5]
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        # Dynamic fusion    
        e1 = self.fusion1(e1, v1)
        e2 = self.fusion2(e2, v2)
        e3 = self.fusion3(e3, v3)
        e4 = self.fusion4(e4, v4)
        # Decoder
        d4 = self.decoder4(self.edge3(e4), e3)
        d3 = self.decoder3(self.edge2(d4), e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)
        return torch.sigmoid(out)
