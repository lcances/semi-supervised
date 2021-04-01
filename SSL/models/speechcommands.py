
# =============================================================================
# UrbanSound8k models
# =============================================================================
from SSL.models.ubs8k import cnn0
from SSL.models.ubs8k_test import cnn03, cnn06

# =============================================================================
#    PyTorch RESNET
# =============================================================================
import torchvision.models as torch_models
from torchvision.models.resnet import Bottleneck, BasicBlock
from SSL.models.wideresnet import ResNet


class mResnet(torch_models.ResNet):
    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

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


# =============================================================================
#    MobileNet
# =============================================================================
from SSL.models.audioset import (
    cnn14,
    MobileNetV1,
    MobileNetV2,
)
