import torch
import argparse
import sys
import os
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, List, Optional

from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.client import PassiveClientManager
from fedlab.core.network import DistNetwork

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
from fednoisy.utils.misc import (
    setup_seed,
    make_dirs,
    make_exp_name,
    result_parser,
    make_alg_name,
    AverageMeter,
)
from fednoisy.utils import misc as misc
from fednoisy.utils.criterion import get_robust_loss, mixup_criterion, loss_coteaching
from fednoisy.utils.mixup import mixup_data
from fednoisy.utils import dynamic_bootstrapping as dynboot


class FedNLLFedAvgClientTrainer(SGDSerialClientTrainer):
    def __init__(
        self,
        model,
        num_clients,
        cuda=True,
        device=None,
        logger=None,
        personal=False,
        args=None,
    ) -> None:
        SGDSerialClientTrainer.__init__(
            self, model, num_clients, cuda, device, logger, personal
        )
        self.cache = []
        self.args = args

    @property
    def model_parameters(self) -> torch.Tensor:
        return misc.serialize_model(self._model)

    def set_model(self, parameters: torch.Tensor):
        misc.deserialize_model(self._model, parameters)

    def setup_optim(self, epochs, batch_size, lr, weight_decay, momentum):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(
            self._model.parameters(), lr, weight_decay=weight_decay, momentum=momentum
        )
        self.criterion = get_robust_loss(CLASS_NUM[self.args.dataset], self.args)

    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list, cur_round):
        self.round = cur_round
        self._LOGGER.info(f"Round {self.round} selected clients: {id_list}")
        model_parameters = payload[0]
        for cid in id_list:
            self.cur_cid = cid
            data_loader = self.dataset.get_dataloader(
                cid=cid, train=True, batch_size=self.batch_size
            )
            pack = self.train(model_parameters, data_loader)
            loss_, acc_ = self.evaluate()
            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local test accuracy: {acc_*100:.2f}%, local test loss: {loss_:.4f}"
            )
            self.cache.append(pack)

    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.train()
        data_size = len(train_loader.dataset)

        for epoch in range(self.epochs):
            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]"
            )
            for imgs, labels, noisy_labels in train_loader:
                if self.cuda:
                    imgs = imgs.cuda(self.device)
                    noisy_labels = noisy_labels.cuda(self.device)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, noisy_labels)

                self.optimizer.zero_grad()
                self._model.zero_grad()
                loss.backward()
                self.optimizer.step()

        local_result = [self.model_parameters, data_size]
        return local_result

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

        return loss_, acc_


class FedNLLFedAvgMixupClientTrainer(FedNLLFedAvgClientTrainer):
    def __init__(
        self,
        model,
        num_clients,
        cuda=True,
        device=None,
        logger=None,
        personal=False,
        args=None,
    ) -> None:
        FedNLLFedAvgClientTrainer.__init__(
            self,
            model,
            num_clients,
            cuda,
            device,
            logger,
            personal,
            args,
        )

    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.train()
        data_size = len(train_loader.dataset)

        for epoch in range(self.epochs):
            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]"
            )
            train_loss = 0
            correct = 0
            total = 0
            batch_num = len(train_loader)
            for batch_idx, (imgs, labels, noisy_labels) in enumerate(train_loader):
                if self.cuda:
                    imgs = imgs.cuda(self.device)
                    noisy_labels = noisy_labels.cuda(self.device)

                imgs, targets_a, targets_b, lmbd = mixup_data(
                    imgs, noisy_labels, self.args.mixup_alpha, self.device
                )

                outputs = self.model(imgs)
                loss = mixup_criterion(
                    self.criterion, outputs, targets_a, targets_b, lmbd
                )

                self.optimizer.zero_grad()
                self._model.zero_grad()
                loss.backward()
                self.optimizer.step()

                # with torch.no_grad():
                #     train_loss += loss.detach()
                #     _, pred = torch.max(outputs.data, 1)
                #     total += noisy_labels.shape[0]
                #     correct += (
                #         lmbd * pred.eq(targets_a.data).cpu().sum().float()
                #         + (1 - lmbd) * pred.eq(targets_b.data).cpu().sum().float()
                #     )
            # avg_train_loss = train_loss / batch_num
            # train_acc = correct / total
            # self._LOGGER.info(
            #     f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}] train accuracy: {train_acc*100:.2f}%; local train loss: {avg_train_loss:.2f}"
            # )

        local_result = [self.model_parameters, data_size]
        return local_result


class FedNLLFedAvgCoteachingClientTrainer(FedNLLFedAvgClientTrainer):
    def __init__(
        self,
        model,
        num_clients,
        cuda=True,
        device=None,
        logger=None,
        personal=False,
        args=None,
    ) -> None:
        FedNLLFedAvgClientTrainer.__init__(
            self,
            model,
            num_clients,
            cuda,
            device,
            logger,
            personal,
            args,
        )

        # ---- initial hyperparameter setting ----
        # TODO: a possible hyperparameter setting for co-teaching in FL
        if args.coteaching_forget_rate is None:
            if args.globalize is True:
                estimate_noise_ratio = args.noise_ratio
            else:
                estimate_noise_ratio = (args.min_noise_ratio + args.max_noise_ratio) / 2
            self.coteaching_forget_rate = [
                estimate_noise_ratio for _ in range(num_clients)
            ]
        else:
            self.coteaching_forget_rate = [
                args.coteaching_forget_rate for _ in range(num_clients)
            ]

        rate_schedule = [
            np.ones(args.com_round) * self.coteaching_forget_rate[cid]
            for cid in range(num_clients)
        ]
        for cid in range(num_clients):
            rate_schedule[cid][: args.coteaching_num_gradual] = np.linspace(
                0,
                self.coteaching_forget_rate[cid] ** args.coteaching_exponent,
                args.coteaching_num_gradual,
            )
        self.rate_schedule = rate_schedule

    def setup_optim(self, epochs, batch_size, lr, weight_decay, momentum):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer1 = torch.optim.SGD(
            self._model.models[0].parameters(),
            lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        self.optimizer2 = torch.optim.SGD(
            self._model.models[1].parameters(),
            lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

    def train(self, model_parameters, train_loader):
        pure_ratio1_ = AverageMeter()
        pure_ratio2_ = AverageMeter()
        acc1_ = AverageMeter()
        acc2_ = AverageMeter()

        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.models[0].train()
        self._model.models[1].train()
        data_size = len(train_loader.dataset)

        for epoch in range(self.epochs):
            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]"
            )
            for iter_idx, (imgs, labels, noisy_labels) in enumerate(train_loader):
                noise_or_not = noisy_labels == labels
                batch_size = len(noisy_labels)
                if self.cuda:
                    imgs = imgs.cuda(self.device)
                    noisy_labels = noisy_labels.cuda(self.device)

                outputs = self.model(imgs)

                with torch.no_grad():
                    _, predicted1 = torch.max(outputs[0], 1)
                    _, predicted2 = torch.max(outputs[1], 1)
                    acc1_.update(
                        torch.sum(predicted1.eq(noisy_labels)).item() / batch_size,
                        batch_size,
                    )
                    acc2_.update(
                        torch.sum(predicted2.eq(noisy_labels)).item() / batch_size,
                        batch_size,
                    )

                loss1, loss2, batch_pure_ratio1, batch_pure_ratio2 = loss_coteaching(
                    outputs[0],
                    outputs[1],
                    noisy_labels,
                    self.rate_schedule[self.cur_cid][self.round],
                    noise_or_not,
                )
                pure_ratio1_.update(100 * batch_pure_ratio1.item(), batch_size)
                pure_ratio2_.update(100 * batch_pure_ratio2.item(), batch_size)

                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                self._model.models[0].zero_grad()
                self._model.models[1].zero_grad()
                loss1.backward()
                loss2.backward()
                self.optimizer1.step()
                self.optimizer2.step()
                self._LOGGER.info(
                    f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}] iter {iter_idx}: "
                    f"loss1: {loss1.item():.4f}, loss2: {loss2.item():.4f}"
                )

        self._LOGGER.info(
            f"Round {self.round} client-{self.cur_cid} local train done: "
            f"train_acc1={acc1_.avg*100:.2f}%, train_acc2={acc2_.avg*100:.2f}%,"
            f"pure_ratios1: {pure_ratio1_.avg:.2f}%, pure_ratios2: {pure_ratio2_.avg:.2f}%"
        )
        local_result = [self.model_parameters, data_size]
        return local_result


class FedNLLFedAvgDynamicBootstrappingClientTrainer(FedNLLFedAvgClientTrainer):
    def __init__(
        self,
        model,
        num_clients,
        cuda=True,
        device=None,
        logger=None,
        personal=False,
        args=None,
    ) -> None:
        FedNLLFedAvgClientTrainer.__init__(
            self,
            model,
            num_clients,
            cuda,
            device,
            logger,
            personal,
            args,
        )

        # ---- initial hyperparameter setting ----
        # TODO: a possible hyperparameter setting for Dynamic Bootstraping in FL
        self.bmm_model = None
        self.bmm_model_maxLoss = 0
        self.bmm_model_minLoss = 0

        self.guidedMixup_round = int(105 / 300 * self.args.com_round)
        if args.dynboot_mixup == "dynamic":
            self.bootstrap_round_mixup = self.guidedMixup_round + int(
                5 / 300 * self.args.com_round
            )
        else:
            self.bootstrap_round_mixup = int(105 / 300 * self.args.com_round)

        self.temp_length = (
            int(200 / 300 * self.args.com_round) - self.bootstrap_round_mixup
        )
        self.temp_vec = np.linspace(1, 0.001, self.temp_length)
        self.k = 0

    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.train()
        data_size = len(train_loader.dataset)

        for epoch in range(self.epochs):
            msg = ""
            first_flag = self.round * self.epochs + epoch == 0
            if self.args.dynboot_mixup == "static":
                alpha = self.args.dynboot_alpha
                if self.round < self.bootstrap_round_mixup:
                    self._LOGGER.info(
                        f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: NORMAL mixup for {self.bootstrap_round_mixup} rounds"
                    )
                    train_loss, train_acc = dynboot.train_mixUp(
                        self._model, train_loader, self.optimizer, 32, self.device
                    )
                else:
                    if self.args.dynboot_bootbeta == "hard":
                        self._LOGGER.info(
                            f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: HARD BETA bootstrapping and NORMAL mixup from round {self.bootstrap_round_mixup+1}"
                        )
                        train_loss, train_acc = dynboot.train_mixUp_HardBootBeta(
                            self._model,
                            train_loader,
                            self.optimizer,
                            self.bmm_model,
                            self.bmm_model_maxLoss,
                            self.bmm_model_minLoss,
                            alpha,
                            self.args.dymboot_reg,
                            CLASS_NUM[self.args.dataset],
                            self.device,
                        )
                    elif self.args.dynboot_bootbeta == "soft":
                        self._LOGGER.info(
                            f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: SOFT BETA bootstrapping and NORMAL mixup from round {self.bootstrap_round_mixup+1}"
                        )
                        train_loss, train_acc = dynboot.train_mixUp_SoftBootBeta(
                            self._model,
                            train_loader,
                            self.optimizer,
                            self.bmm_model,
                            self.bmm_model_maxLoss,
                            self.bmm_model_minLoss,
                            alpha,
                            self.args.dymboot_reg,
                            first_flag,
                            self.device,
                        )

            if self.args.dynboot_mixup == "dynamic":
                alpha = self.args.dynboot_alpha
                if self.round < self.guidedMixup_round:
                    self._LOGGER.info(
                        f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: NORMAL mixup for {self.guidedMixup_round} rounds"
                    )
                    train_loss, train_acc = dynboot.train_mixUp(
                        self._model, train_loader, self.optimizer, alpha, self.device
                    )
                elif self.round < self.bootstrap_round_mixup:
                    self._LOGGER.info(
                        f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: Dynamic mixup from {self.guidedMixup_round} rounds"
                    )
                    train_loss, train_acc = dynboot.train_mixUp_Beta(
                        self.model,
                        train_loader,
                        self.optimizer,
                        self.bmm_model,
                        self.bmm_model_maxLoss,
                        self.bmm_model_minLoss,
                        alpha,
                        first_flag,
                        self.device,
                    )
                else:
                    self._LOGGER.info(
                        f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: Going from SOFT BETA bootstrapping to HARD BETA with linear temperature and Dynamic mixup from {self.bootstrap_round_mixup} rounds"
                    )
                    k = min(
                        self.round - self.bootstrap_round_mixup,
                        self.temp_length - 1,
                    )
                    Temp = self.temp_vec[k]
                    train_loss, train_acc = dynboot.train_mixUp_SoftHardBetaDouble(
                        self._model,
                        train_loader,
                        self.optimizer,
                        self.bmm_model,
                        self.bmm_model_maxLoss,
                        self.bmm_model_minLoss,
                        alpha,
                        self.args.dynboot_reg,
                        first_flag,
                        Temp,
                        CLASS_NUM[self.args.dataset],
                        self.device,
                    )
                    msg = f"Temperature: {Temp:.4f}"

            # training tracking loss
            (
                self.bmm_model,
                self.bmm_model_maxLoss,
                self.bmm_model_minLoss,
            ) = dynboot.track_training_loss(self.model, train_loader, self.device)

            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: train_acc: {train_acc*100:.2f}%, train_loss: {train_loss:.4f}, {msg}"
            )

        local_result = [self.model_parameters, data_size]
        return local_result
