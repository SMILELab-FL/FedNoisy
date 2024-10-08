from .cnn import Cifar10Net, SimpleCNN, SimpleCNNMNIST, ConvNet5
from .lenet import CIFAR10LeNet, MNISTLeNet
from .toymodel import ToyModel
from .resnet import ResNet18, ResNet34
from .resnet2 import ResNet20, ResNet32
from .preresnet import ResNet18 as PreResNet18
from .wideresnet import WRN28_10, WRN40_2
from .inceptionresnetV2 import InceptionResNetV2
from .vgg import VGG11, VGG13, VGG16, VGG19
from .fcnet import TwoLayerLinearNet, LinearRegression
from .mlp import DMLP

from torch import nn
import torchvision


def build_model(
    model_name: str,
    num_classes: int = 10,
    dataset: str = "CIFAR10",
    input_size: int = 20,
):
    """_summary_

    Args:
        model_name (str): Can be
        num_classes (int, optional): Number of classes. Defaults to 10.
        dataset (str, optional): _description_. Defaults to 'CIFAR10'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if model_name == "TwoLayerLinear":
        if dataset.upper() == "MNIST":
            base_model = TwoLayerLinearNet(
                input_size=784, hidden_size=500, num_classes=num_classes
            )
    elif model_name == "LinearRegression":
        base_model = LinearRegression(input_dim=input_size, output_dim=1)
    elif model_name == "Cifar10Net":
        base_model = Cifar10Net()
    elif model_name == "ConvNet5":
        if dataset.upper() == "MNIST":
            base_model = ConvNet5(num_channels=1)
        else:
            raise ValueError(
                f"{model_name} currently not supports {dataset}. Only mnist is supported."
            )
    elif model_name == "SimpleCNN":
        if dataset.upper() in ["CIFAR10", "SVHN"]:
            base_model = SimpleCNN(
                input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10
            )
        elif dataset.upper() in ["MNIST", "FEMNIST", "FMNIST"]:
            base_model = SimpleCNNMNIST(
                input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10
            )
    elif model_name == "DMLP":
        if dataset.upper() == "MNIST":
            data_shape = (1, 28, 28)
        elif dataset.upper() == "CIFAR10":
            data_shape = (3, 32, 32)
        else:
            raise ValueError(
                f"{model_name} currently not supports {dataset}. Only cifar10, mnist are supported."
            )
        base_model = DMLP(num_classes=num_classes, data_shape=data_shape)
    elif model_name == "LeNet":
        if dataset.upper() in ["CIFAR10", "SVHN"]:
            base_model = CIFAR10LeNet()
        elif dataset.upper() == "MNIST":
            base_model = MNISTLeNet()
        else:
            raise ValueError(
                f"{model_name} currently does not support {dataset.upper()}. Only CIFAR10 and MNIST are supported."
            )
    elif model_name == "PreResNet18":
        base_model = PreResNet18(num_classes)
    elif model_name == "ResNet18":
        base_model = ResNet18(num_classes)
    elif model_name == "ResNet20":
        base_model = ResNet20(num_classes)
    elif model_name == "ResNet32":
        base_model = ResNet32(num_classes)
    elif model_name == "ResNet34":
        base_model = ResNet34(num_classes)
    elif model_name == "ResNet50":
        base_model = torchvision.models.resnet50(pretrained=True)
        base_model.fc = nn.Linear(2048, num_classes)
    elif model_name == "ToyModel":
        base_model = ToyModel(type=dataset.upper())
    elif model_name == "WRN28_10":
        base_model = WRN28_10(num_classes)
    elif model_name == "WRN40_2":
        base_model = WRN40_2(num_classes)
    elif model_name == "VGG11":
        base_model = VGG11()
    elif model_name == "VGG13":
        base_model = VGG13()
    elif model_name == "VGG16":
        base_model = VGG16()
    elif model_name == "VGG19":
        base_model = VGG19()
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unrecognized model: {model_name}. Currently only support 'Cifar10Net', 'SimpleCNN',  'LeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'ToyModel', 'ResNet18', 'ResNet20', 'WRN28_10', 'WRN40_2', 'ResNet32', 'ResNet34', and 'InceptionResNetV2'."
        )

    return base_model


def build_multi_model(
    model_name: str, num_classes: int = 10, dataset: str = "CIFAR10", num_models=2
):
    """_summary_

    Args:
        model_name (str): Can be
        num_classes (int, optional): Number of classes. Defaults to 10.
        dataset (str, optional): _description_. Defaults to 'CIFAR10'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if model_name == "TwoLayerLinear":
        if dataset.upper() == "MNIST":
            base_model = MultiModel(
                [
                    TwoLayerLinearNet(
                        input_size=28 * 28, hidden_size=500, num_classes=10
                    )
                    for _ in range(num_models)
                ]
            )
    elif model_name == "Cifar10Net":
        base_model = MultiModel([Cifar10Net() for _ in range(num_models)])
    elif model_name == "SimpleCNN":
        if dataset.upper() in ["CIFAR10", "SVHN"]:
            base_model = MultiModel(
                [
                    SimpleCNN(
                        input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10
                    )
                    for _ in range(num_models)
                ]
            )

        elif dataset.upper() in ["MNIST", "FEMNIST", "FMNIST"]:
            base_model = MultiModel(
                [
                    SimpleCNNMNIST(
                        input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10
                    )
                    for _ in range(num_models)
                ]
            )
    elif model_name == "LeNet":
        if dataset.upper() in ["CIFAR10", "SVHN"]:
            model_arch = CIFAR10LeNet
        elif dataset.upper() == "MNIST":
            model_arch = MNISTLeNet
        else:
            raise ValueError(
                f"{model_name} currently does not support {dataset.upper()}. Only CIFAR10 and MNIST are supported."
            )
        base_model = MultiModel([model_arch() for _ in range(num_models)])
    elif model_name == "ResNet18":
        base_model = MultiModel([ResNet18(num_classes) for _ in range(num_models)])
    elif model_name == "ResNet34":
        base_model = MultiModel([ResNet34(num_classes) for _ in range(num_models)])
    elif model_name == "ResNet50":
        net1 = torchvision.models.resnet50(pretrained=True)
        net1.fc = nn.Linear(2048, num_classes)
        net2 = torchvision.models.resnet50(pretrained=True)
        net2.fc = nn.Linear(2048, num_classes)
        base_model = MultiModel([net1, net2])
    elif model_name == "ToyModel":
        base_model = MultiModel(
            [ToyModel(type=dataset.upper()) for _ in range(num_models)]
        )
    elif model_name == "WRN28_10":
        base_model = MultiModel([WRN28_10(num_classes) for _ in range(num_models)])
    elif model_name == "WRN40_2":
        base_model = MultiModel([WRN40_2(num_classes) for _ in range(num_models)])
    elif model_name == "VGG11":
        base_model = MultiModel([VGG11() for _ in range(num_models)])
    elif model_name == "VGG13":
        base_model = MultiModel([VGG13() for _ in range(num_models)])
    elif model_name == "VGG16":
        base_model = MultiModel([VGG16() for _ in range(num_models)])
    elif model_name == "VGG19":
        base_model = MultiModel([VGG19() for _ in range(num_models)])
    elif model_name == "InceptionResNetV2":
        base_model = MultiModel(
            [InceptionResNetV2(num_classes=num_classes) for _ in range(num_models)]
        )
    else:
        raise ValueError(
            f"Unrecognized model: {model_name}. Currently only support 'Cifar10Net', 'SimpleCNN',  'LeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'ToyModel', 'ResNet18', 'ResNet20', 'WRN28_10', 'WRN40_2', 'ResNet32', 'ResNet34', and 'InceptionResNetV2'."
        )

    return base_model


class MultiModel(nn.Module):
    def __init__(self, init_models) -> None:
        super().__init__()
        self.models = nn.ModuleList(init_models)

    def forward(self, x):
        outputs = [net(x) for net in self.models]
        return outputs

    def get_model_list(self):
        return [model for model in self.models]
