from ..models import *

class ModelFactory:
    @staticmethod
    def get_model(model_name):
        name = model_name.lower()
        if name == "lenet":
            return LeNet()
        if name == "densenet121":
            return DenseNet121()
        if name == "densenet169":
            return DenseNet169()
        if name == "densenet201":
            return DenseNet201()
        if name == "densenet161":
            return DenseNet161()
        if name == "densenet_cifar":
            return densenet_cifar()
        if name == "resnet18":
            return ResNet18()
        if name == "resnet34":
            return ResNet34()
        if name == "resnet50":
            return ResNet50()
        if name == "resnet101":
            return ResNet101()
        if name == "resnet152":
            return ResNet152()
        if name == "resnext29_2x64d":
            return ResNeXt29_2x64d()
        if name == "resnext29_4x64d":
            return ResNeXt29_4x64d()
        if name == "resnext29_8x64d":
            return ResNeXt29_8x64d()
        if name == "resnext29_32x4d":
            return ResNeXt29_32x4d()
