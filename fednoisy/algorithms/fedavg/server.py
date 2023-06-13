import sys
import argparse
import os
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from fedlab.core.server.manager import SynchronousServerManager

from fedlab.core.client.trainer import SerialClientTrainer
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.core.network import DistNetwork
from fedlab.utils import Logger, Aggregators, SerializationTool

sys.path.append(os.getcwd())
from fednoisy.data.NLLData import functional as nllF
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    NORM_VALUES,
)

from fednoisy.utils.misc import AverageMeter
from fednoisy.utils import misc as misc


class FedAvgServerHandler(SyncServerHandler):
    def __init__(
        self,
        model: torch.nn.Module,
        global_round: int,
        sample_ratio: float,
        nll_name: str = None,
        cuda: bool = True,
        device: str = None,
        logger: Logger = None,
        args=None,
    ):
        SyncServerHandler.__init__(
            self, model, global_round, sample_ratio, cuda, device, logger
        )
        self.nll_name = nll_name
        self.args = args

    @property
    def model_parameters(self) -> torch.Tensor:
        return misc.serialize_model(self._model)

    def set_model(self, parameters: torch.Tensor):
        misc.deserialize_model(self._model, parameters)

    def setup_dataset(self, dataset) -> None:
        self.dataset = dataset

    def global_update(self, buffer):
        parameters_list = [elem[0] for elem in buffer]
        weights = [elem[1] for elem in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        self.set_model(serialized_parameters)
        self._LOGGER.info(
            f"Round [{self.round}/{self.global_round}] server global update done."
        )

    def evaluate(self):
        self._model.eval()
        test_loader = self.dataset.get_dataloader(train=False, batch_size=128)
        multimodel = hasattr(self._model, "models")
        loss_, acc_ = misc.evaluate(
            self._model,
            nn.CrossEntropyLoss(),
            test_loader,
            self.device,
            multimodel=multimodel,
        )

        return loss_, acc_
