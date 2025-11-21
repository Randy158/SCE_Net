import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from model.resnet import create_resnet
from functools import partial
from model.Mamba import VSSE
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
        
        # Initialize ResNet34 backbone
        resnet_backbone = create_resnet('resnet34', pretrain=True)
        self.initial_conv = resnet_backbone.conv1
        self.initial_bn = resnet_backbone.bn1
        self.initial_relu = resnet_backbone.relu
        self.initial_maxpool = resnet_backbone.maxpool
       
        self.encoder_stage1 = resnet_backbone.layer1
        self.encoder_stage2 = resnet_backbone.layer2
        self.encoder_stage3 = resnet_backbone.layer3
        self.encoder_stage4 = resnet_backbone.layer4
    
        self.decoder_stage4 = DecoderBlock(512, 256, 256)  
        self.decoder_stage3 = DecoderBlock(256, 128, 128)  
        self.decoder_stage2 = DecoderBlock(128, 64, 64)  
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.final_upconv1  = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nonlinearity
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nonlinearity
        self.final_conv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        
        self.conv = nn.Conv2d(1, 3, kernel_size=1, stride=1)
        self.vssm_encoder = VSSE(patch_size=2, in_chans=48)
        # initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(48, eps=1e-5, affine=True),
        )

        self.fusion_stage1 = DynamicFusion(64)
        self.fusion_stage2 = DynamicFusion(128)
        self.fusion_stage3 = DynamicFusion(256)
        self.fusion_stage4 = DynamicFusion(512)
 
        self.edge_attention3 = EdgeAttention(512)
        self.edge_attention2 = EdgeAttention(256)
   
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
        stem_output = self.stem(x)  # [B, 48, H/2, W/2]
        # Before VSS
        stem_output = self.vss_pre_cnn(stem_output)  
        # VSSM Encoder
        vssm_features = self.vssm_encoder(stem_output)  # [B, C, H, W]
        # After VSS
        vssm_features = [self.vss_post_cnn[i](vssm_features[i]) for i in range(5)]
        v0, v1, v2, v3, v4 = vssm_features[0:5]
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = self.initial_maxpool(x)
        e1 = self.encoder_stage1(x)
        e2 = self.encoder_stage2(e1)
        e3 = self.encoder_stage3(e2)
        e4 = self.encoder_stage4(e3)
        
        # Dynamic fusion    
        e1 = self.fusion_stage1(e1, v1)
        e2 = self.fusion_stage2(e2, v2)
        e3 = self.fusion_stage3(e3, v3)
        e4 = self.fusion_stage4(e4, v4)
        # Decoder
        d4 = self.decoder_stage4(self.edge3(e4), e3)
        d3 = self.decoder_stage3(self.edge2(d4), e2)
        d2 = self.decoder_stage2(d3, e1)
        d1 = self.decoder_stage1(d2)
        
        out = self.final_upconv1(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)
        return torch.sigmoid(out)
