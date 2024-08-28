import torch
import argparse
import sys
import os
from copy import deepcopy
from typing import Dict, Tuple, List, Optional

from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fedlab.contrib.dataset.basic_dataset import FedDataset
from fedlab.utils.logger import Logger

sys.path.append(os.getcwd())
from fednoisy.data.NLLData import functional as nllF
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    NORM_VALUES,
)

# from fednoisy.utils import misc as misc
from fednoisy.data.NLLData import functional as nllF
from fednoisy.data.dataset import FedNLLClientDataset
from fednoisy.utils.misc import setup_seed, make_dirs, make_exp_name, AverageMeter
from fednoisy.models.build_model import build_model
from fednoisy.algorithms.singleset.misc import read_singlenll_args


args = read_singlenll_args()
if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

setup_seed(args.seed)
nll_name = nllF.FedNLL_name(**vars(args))

# ==== Data loader
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(*NORM_VALUES[args.dataset]),
    ]
)
test_dataset = torchvision.datasets.CIFAR10(
    train=False, root=args.raw_data_dir, transform=test_transform
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=False
)
dataset = FedNLLClientDataset(args)
train_loader = dataset.get_dataloader(args.client_id, args.batch_size)

# ==== Get model
model = build_model(args.model, CLASS_NUM[args.dataset])
model = model.to(args.device)

# === Optimizer
# optimizer = torch.optim.SGD(
#     model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum
# )
optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()

# ==== Setup log
logger = Logger(log_name="SingleSetTrainer")

# ==== Training
for epoch in range(args.epochs):
    logger.info(f"Epoch [{epoch}/{args.epochs}] Client-{args.client_id} local training")
    model.train()
    for imgs, labels, noisy_labels in train_loader:
        if args.cuda:
            imgs = imgs.cuda(args.device)
            noisy_labels = noisy_labels.cuda(args.device)

        output = model(imgs)
        loss = criterion(output, noisy_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # logger.info(f"loss: {loss.item()}")

    loss_, acc_ = evaluate(model, nn.CrossEntropyLoss(), test_loader)
    logger.info(
        f"Epoch [{epoch}/{args.epochs}] Client-{args.client_id} test accuracy: {acc_*100}%"
    )
