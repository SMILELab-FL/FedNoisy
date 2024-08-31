import sys
import argparse
import os
import random
import numpy as np
from collections import Counter
from copy import deepcopy

from sklearn.mixture import GaussianMixture

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
from fednoisy.utils import fednoro as fednoro


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
        # self.nll_name = nll_name
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

        if self.args.dataset == "webvision":
            imagenet_test_loader = self.dataset.get_dataloader(
                train=False, batch_size=128, imagenet=True
            )

            imagenet_loss_, imagenet_acc1_, imagenet_acc5_ = misc.evaluate(
                self._model,
                nn.CrossEntropyLoss(),
                imagenet_test_loader,
                self.device,
                multimodel=multimodel,
                imagenet=True,
            )
            return (
                loss_,
                acc_,
                imagenet_loss_,
                imagenet_acc1_,
                imagenet_acc5_,
            )

        return loss_, acc_


class FedNoRoServerHandler(SyncServerHandler):
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
        # self.nll_name = nll_name
        self.args = args
        self.nll_name = nllF.FedNLL_name(**vars(args))
        nll_filename = f"{self.nll_name}_seed_{args.seed}_setting.pt"
        nll_file_path = os.path.join(args.data_dir, nll_filename)
        self.client_true_noise_ratio = torch.load(nll_file_path)["true_noise_ratio"]

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
        sampled_clients = [elem[2] for elem in buffer]
        if self.round < self.args.fednoro_warmup:
            # warmup stage using FedAvg
            serialized_parameters = Aggregators.fedavg_aggregate(
                parameters_list, weights
            )
        else:
            # stage 2 aggregation based on distance
            serialized_parameters = fednoro.partialDaAgg(
                parameters_list,
                weights,
                sampled_clients,
                self.clean_clients,
                self.noisy_clients,
            )

        self.set_model(serialized_parameters)
        self._LOGGER.info(
            f"Round [{self.round}/{self.global_round}] server global update done."
        )

    def evaluate(self):
        test_loader = self.dataset.get_dataloader(train=False, batch_size=128)
        multimodel = hasattr(self._model, "models")
        loss_, acc_ = misc.evaluate(
            self._model,
            nn.CrossEntropyLoss(),
            test_loader,
            self.device,
            multimodel=multimodel,
        )

        if self.args.dataset == "webvision":
            imagenet_test_loader = self.dataset.get_dataloader(
                train=False, batch_size=128, imagenet=True
            )

            imagenet_loss_, imagenet_acc1_, imagenet_acc5_ = misc.evaluate(
                self._model,
                nn.CrossEntropyLoss(),
                imagenet_test_loader,
                self.device,
                multimodel=multimodel,
                imagenet=True,
            )
            return (
                loss_,
                acc_,
                imagenet_loss_,
                imagenet_acc1_,
                imagenet_acc5_,
            )

        return loss_, acc_

    def justify_noisy_client(self):
        num_classes = CLASS_NUM[self.args.dataset]
        num_clients = self.args.num_clients
        metrics = np.zeros((num_clients, num_classes)).astype("float")
        num = np.zeros((num_clients, num_classes)).astype("float")
        criterion = nn.CrossEntropyLoss(reduction="none")
        # calculate per client per class loss for each client
        for cid in range(num_clients):
            local_train_dataset = self.dataset.get_dataset(cid=cid, train=True)
            local_train_loader = DataLoader(
                local_train_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            cur_noisy_labels = np.array(local_train_dataset.noisy_labels)
            local_class_num_cnt = Counter(local_train_dataset.noisy_labels)
            local_output, loss = fednoro.get_output(
                self._model,
                local_train_loader,
                softmax=False,
                criterion=criterion,
                device=self.device,
            )

            for cc in range(num_classes):
                num[cid, cc] += local_class_num_cnt.get(cc, 0)
                cur_cc_idxs = np.where(cur_noisy_labels == cc)
                metrics[cid, cc] += loss[cur_cc_idxs[0]].sum().item()

        metrics = metrics / num
        for i in range(metrics.shape[0]):
            for j in range(metrics.shape[1]):
                if np.isnan(metrics[i, j]):
                    metrics[i, j] = np.nanmin(metrics[:, j])
        for j in range(metrics.shape[1]):
            metrics[:, j] = (metrics[:, j] - metrics[:, j].min()) / (
                metrics[:, j].max() - metrics[:, j].min()
            )

        self._LOGGER.info("metrics:")
        self._LOGGER.info(metrics)

        vote = []
        for i in range(9):
            gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics)
            gmm_pred = gmm.predict(metrics)
            noisy_clients = np.where(gmm_pred == np.argmax(gmm.means_.sum(1)))[0]
            noisy_clients = set(list(noisy_clients))
            vote.append(noisy_clients)
        cnt = []
        for i in vote:
            cnt.append(vote.count(i))
        noisy_clients = list(vote[cnt.index(max(cnt))])

        if self.client_true_noise_ratio[0] is not None:
            real_noisy_clients = [
                cid
                for cid in range(num_clients)
                if self.client_true_noise_ratio[cid] > 0.0
            ]
        else:
            real_noisy_clients = list(range(num_clients))

        self._LOGGER.info(
            f"selected noisy clients: {noisy_clients}, real noisy clients: {real_noisy_clients}"
        )
        clean_clients = list(set(list(range(num_clients))) - set(noisy_clients))
        self._LOGGER.info(f"selected clean clients: {clean_clients}")

        self.noisy_clients = noisy_clients
        self.clean_clients = clean_clients
        return clean_clients, noisy_clients
