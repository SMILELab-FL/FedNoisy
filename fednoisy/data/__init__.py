import torchvision.transforms as transforms

# automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
# CIFAR10_TRANSITION_MATRIX = {0: 0, 1: 1, 2: 0, 3: 5, 4: 7, 5: 3, 6: 6, 7: 7, 8: 8, 9: 1}


NORM_VALUES = {
    "cifar10": [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
    "cifar100": [(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)],
    "mnist": [(0.1307,), (0.3081,)],
    "svhn": [(0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)],
    "clothing1m": [(0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)],
    "webvision": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
}

TRAIN_TRANSFORM = {
    "cifar10": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES["cifar10"]),
        ]
    ),
    "cifar100": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES["cifar100"]),
        ]
    ),
    "mnist": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES["mnist"]),
        ]
    ),
    "svhn": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES["svhn"]),
        ]
    ),
}

TEST_TRANSFORM = {
    "cifar10": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES["cifar10"]),
        ]
    ),
    "cifar100": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES["cifar100"]),
        ]
    ),
    "mnist": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES["mnist"]),
        ]
    ),
    "svhn": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES["svhn"]),
        ]
    ),
}


TRAIN_SAMPLE_NUM = {"cifar10": 50000, "cifar100": 50000, "mnist": 60000, "svhn": 73257}

TEST_SAMPLE_NUM = {"cifar10": 10000, "cifar100": 10000, "mnist": 10000, "svhn": 26032}

CLASS_NUM = {
    "cifar10": 10,
    "cifar100": 100,
    "mnist": 10,
    "svhn": 10,
    "clothing1m": 14,
    "webvision": 50,
}

from .NLLData.functional import TRANSITION_MATRIX
