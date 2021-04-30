import numpy as np

import torch.nn as nn
import librosa

from SSL.layers import ConvPoolReLU, ConvReLU, ConvBNReLUPool, ConvAdvBNReLUPool


# =============================================================================
#    MobileNet
# =============================================================================
from SSL.models.audioset import (
    cnn14,
    MobileNetV1,
    MobileNetV2,
)

class cnn(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.features = nn.Sequential(
            ConvPoolReLU(1, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.0),
            ConvPoolReLU(32, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.3),
            ConvPoolReLU(32, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.3),
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.ReLU6(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1696, 10) # TODO fill
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x


class cnn0(nn.Module):
    def __init__(self, **kwargs):
        super(cnn0, self).__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 24, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(24, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(48, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvReLU(48, 48, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1008, 10),
            #             nn.ReLU(inplace=True),
            #             nn.Dropout(0.5),
            #             nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x


class cnn_advBN(nn.Module):
    """
    Basic CNN model with adversarial dedicated Batch Normalization

    """
    def __init__(self, *kwargs):
        super(cnn_advBN, self).__init__()

        self.features = nn.Sequential(
            ConvAdvBNReLUPool(1, 24, 3, 1, 1, (4, 2), (4, 2)),
            ConvAdvBNReLUPool(24, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvAdvBNReLUPool(48, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvReLU(48, 48, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1008, 10),
            #             nn.ReLU(inplace=True),
            #             nn.Dropout(0.5),
            #             nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x



import torchvision.models as torch_models
from torchvision.models.resnet import Bottleneck, BasicBlock
from SSL.models.wideresnet import ResNet


class mResnet(torch_models.ResNet):
    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])
        x = x.repeat(1, 3, 1, 1)

        return self._forward_impl(x)


def resnet50(**kwargs):
    num_classes = kwargs.get("num_classes", 10)
    return mResnet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet34(**kwargs):
    num_classes = kwargs.get("num_classes", 10)
    return mResnet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet18(**kwargs):
    num_classes = kwargs.get("num_classes", 10)
    return mResnet(BasicBlock, [2, 2, 2, 2], num_classes)


class mWideResnet(ResNet):
    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])
#         x = x.repeat(1, 3, 1, 1)

        return self._forward_impl(x)


def wideresnet28_2(**kwargs):
    num_classes = kwargs.get("num_classes", 10)
    return mWideResnet([4, 4, 4], num_classes=num_classes)


def wideresnet28_4(**kwargs):
    num_classes = kwargs.get("num_classes", 10)
    return mWideResnet([4, 4, 4], num_classes=num_classes, width=4)


def wideresnet28_8(**kwargs):
    num_classes = kwargs.get("num_classes", 10)
    return mWideResnet([4, 4, 4], num_classes=num_classes, width=8)
