import random
import codecs
import gzip
import lzma
import numpy as np
from PIL import Image
import json
import os
import sys
import torch
from torchnet.meter import AUCMeter

from typing import Dict, Tuple, List, Optional, Union, IO, Callable

from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
)
from fednoisy import visual
from fednoisy.data.NLLData import functional as F
from fednoisy.data.NLLData.BaseNLL import NLLBase
from fednoisy.utils.misc import make_dirs

"""This code is based on torch=1.8.0 and torchvision=0.9.0. The latest version have some differences in torchvision.datasets.MNIST.
"""


class NLLMNIST(NLLBase):
    dataset_name = "mnist"
    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    num_classes = CLASS_NUM[dataset_name]
    train_sample_num = TRAIN_SAMPLE_NUM[dataset_name]
    test_sample_num = TEST_SAMPLE_NUM[dataset_name]
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]
    trainset_filename = f"{dataset_name}-trainset.pt"
    testset_filename = f"{dataset_name}-testset.pt"
    # centralized = True

    def __init__(self, root_dir: str, noise_mode: str, out_dir: str) -> None:
        NLLBase.__init__(self, root_dir, noise_mode, out_dir)
        self.root_dir = os.path.join(root_dir, "raw")

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        self._load_testset()
        self._load_trainset()

    def _load_trainset(self, save: bool = True):
        if os.path.exists(self.trainset_path):
            entry = torch.load(self.trainset_path)
            train_data = entry["data"]
            train_labels = entry["labels"]
        else:
            # process raw train dataset
            img_file = os.path.join(self.root_dir, self.resources[0][0][:-3])
            label_file = os.path.join(self.root_dir, self.resources[1][0][:-3])
            train_data = read_image_file(img_file).numpy()
            train_labels = read_label_file(label_file).tolist()

        self.train_labels = train_labels
        self.train_data = train_data
        print(f"{self.dataset_name} trainset is loaded.")
        if save is True:
            self.save_trainset()

    def _load_testset(self, save: bool = True):
        if os.path.exists(self.testset_path):
            entry = torch.load(self.testset_path)
            test_data = entry["data"]
            test_labels = entry["labels"]
        else:
            # process raw test dataset
            img_file = os.path.join(self.root_dir, self.resources[2][0][:-3])
            label_file = os.path.join(self.root_dir, self.resources[3][0][:-3])
            test_data = read_image_file(img_file).numpy()
            test_labels = read_label_file(label_file).tolist()

        self.test_labels = test_labels
        self.test_data = test_data
        print(f"{self.dataset_name} testset is loaded.")
        if save is True:
            self.save_testset()


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


def open_maybe_compressed_file(path: Union[str, IO]) -> Union[IO, gzip.GzipFile]:
    """Return a file object that possibly decompresses 'path' on the fly.
    Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    if path.endswith(".xz"):
        return lzma.open(path, "rb")
    return open(path, "rb")


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype(">i2"), "i2"),
    12: (torch.int32, np.dtype(">i4"), "i4"),
    13: (torch.float32, np.dtype(">f4"), "f4"),
    14: (torch.float64, np.dtype(">f8"), "f8"),
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 1
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 3
    return x
