# *coding:utf-8 *
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

def conv3x3(in_channels, out_channels, stride=1):
    """
    Define a 3x3 convolutional layer with padding
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResidualBlock(nn.Module):
    """
    Basic ResNet residual block with two 3x3 convolution layers.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class DeepBottleneck(nn.Module):
    """
    High-capacity Bottleneck module using 1x1, 3x3, and 1x1 convolutions to increase model capacity.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(DeepBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion, momentum=BN_MOMENTUM)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# Pre-trained model URLs for different ResNet versions
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNetBuilder(nn.Module):
    """
    A class for building a ResNet model with configurable depths and block types.
    """
    def __init__(self, block, layers):
        super(ResNetBuilder, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Create a sequence of layers for a particular ResNet block.
        """
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        stage_1_feature = x
        x = self.layer2(x)
        stage_2_feature = x
        x = self.layer3(x)
        stage_3_feature = x
        x = self.layer4(x)
        stage_4_feature = x

        return x

    def load_pretrained_weights(self, model_name):
        """
        Load pre-trained weights for the given model.
        """
        url = model_urls[model_name]
        pretrained_state_dict = model_zoo.load_url(url)
        print(f"=> Loading pretrained model from {url}")
        self.load_state_dict(pretrained_state_dict, strict=False)


# Function to create different ResNet models based on the architecture and depth
def create_resnet(model_type, pretrain=True):
    """
    Create a ResNet model based on the specified model type, with an option to load pre-trained weights.
    """
    architectures = {
        'resnet18': (ResidualBlock, [2, 2, 2, 2]),
        'resnet34': (ResidualBlock, [3, 4, 6, 3]),
        'resnet50': (DeepBottleneck, [3, 4, 6, 3]),
        'resnet101': (DeepBottleneck, [3, 4, 23, 3]),
        'resnet152': (DeepBottleneck, [3, 8, 36, 3])
    }

    block, layers = architectures.get(model_type)
    model = ResNetBuilder(block, layers)

    if pretrain:
        model.load_pretrained_weights(model_type)
    
    return model


# Main execution block
if __name__ == '__main__':
    model_name = 'resnet34'  # You can change this to 'resnet18', 'resnet50', 'resnet101', etc.
    model = create_resnet(model_name, pretrain=True)

    print(f"Created model: {model_name}")
    print(model)
