from json import load
import os
import sys
import argparse
import random
from copy import deepcopy
import numpy as np

import wandb

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.core.standalone import StandalonePipeline

sys.path.append(os.getcwd())
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    NORM_VALUES,
)
from fednoisy.data.NLLData import functional as nllF
from fednoisy.algorithms.fedavg.misc import read_fednll_args
from fednoisy.data.dataset import FedNLLDataset
from fednoisy.utils.misc import (
    setup_seed,
    make_dirs,
    make_exp_name,
    result_parser,
    make_alg_name,
)
from fednoisy.models.build_model import build_model


class FedAvgStandalone(StandalonePipeline):
    def __init__(
        self, handler, trainer, args, logger=None, save_best=False, save_last=True
    ):
        super().__init__(handler, trainer)
        self._LOGGER = Logger() if logger is None else logger
        self.save_best = save_best
        self.save_last = save_last
        self.args = args
        self.exp_name = make_exp_name("fedavg", args)
        self.nll_name = nllF.FedNLL_name(**vars(args))
        alg_name = make_alg_name(args)
        self.out_path = os.path.join(
            args.out_dir, self.nll_name, alg_name, self.exp_name
        )
        make_dirs(self.out_path)
        self.record_file = os.path.join(self.out_path, "result_record.txt")
        self.best_model_path = os.path.join(self.out_path, "best_global_model.pth")
        self.last_model_path = os.path.join(self.out_path, "last_global_model.pth")

        self.loss_hist = []
        self.acc_hist = []
        self.max_acc = 0

        self.run_name = f"{self.nll_name}-{alg_name}-{self.exp_name}"
        self.wb_run = wandb.init(
            config=self.args, project=self.args.proj_name, name=self.run_name
        )

    def main(self):
        # check existence of record file
        if os.path.exists(self.record_file):
            accs, _, _ = result_parser(self.record_file)
            if len(accs) >= self.args.com_round:
                self.handler._LOGGER.info(
                    f"Experiment done! Result saved in {self.record_file}!"
                )
                return

        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package

            # client side
            self.trainer.local_process(broadcast, sampled_clients, self.handler.round)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            # evaluate
            self.evaluate()

        if self.save_last:
            torch.save(
                {
                    "model": self.handler._model.state_dict(),
                    "rounds": self.args.com_round,
                },
                self.last_model_path,
            )

        self.wb_run.log({"final_test_acc": np.mean(self.acc_hist[-10:])})
        self.wb_run.finish()

    def evaluate(self):
        if self.args.dataset == "webvision":
            loss_, acc_, imagenet_loss_, imagenet_acc1_, imagenet_acc5_ = (
                self.handler.evaluate()
            )
            self.handler._LOGGER.info(
                f"Round [{self.handler.round - 1}/{self.handler.global_round}] test performance on server: \t Loss: {loss_:.5f} \t Acc: {100*acc_:.3f}% \t ImageNet Loss: {imagenet_loss_:.5f} \t ImageNet Acc1: {100*imagenet_acc1_:.3f}% \t ImageNet Acc5: {100*imagenet_acc5_:.3f}%"
            )
            self.wb_run.log(
                {
                    "test_loss": loss_,
                    "test_acc": 100 * acc_,
                    "test_ImageNet_loss": imagenet_loss_,
                    "test_ImageNet_top1_acc": 100 * imagenet_acc1_,
                    "test_ImageNet_top5_acc": 100 * imagenet_acc5_,
                }
            )
        else:
            loss_, acc_ = self.handler.evaluate()
            self.handler._LOGGER.info(
                f"Round [{self.handler.round - 1}/{self.handler.global_round}] test performance on server: \t Loss: {loss_:.5f} \t Acc: {100*acc_:.3f}%"
            )
            self.wb_run.log({"test_loss": loss_, "test_acc": 100 * acc_})
        self.loss_hist.append(loss_)
        self.acc_hist.append(acc_ * 100)
        record = open(self.record_file, "w")
        record.write(f"{vars(self.args)}\n")
        record.write("acc:" + str(self.acc_hist) + "\n")
        record.write("loss:" + str(self.loss_hist) + "\n")
        record.close()

        if self.save_best:
            if acc_ > self.max_acc:
                self.max_acc = acc_
                torch.save(self.handler._model.state_dict(), self.best_model_path)
                self.handler._LOGGER.info(f"Best global model saved.")


class FedNoRoStandalone(FedAvgStandalone):
    def __init__(
        self, handler, trainer, args, logger=None, save_best=False, save_last=True
    ):
        StandalonePipeline.__init__(self, handler, trainer)
        self._LOGGER = Logger() if logger is None else logger
        self.save_best = save_best
        self.save_last = save_last
        self.args = args
        self.exp_name = make_exp_name("fednoro", args)
        self.nll_name = nllF.FedNLL_name(**vars(args))
        alg_name = make_alg_name(args)
        self.out_path = os.path.join(
            args.out_dir, self.nll_name, alg_name, self.exp_name
        )
        make_dirs(self.out_path)
        self.record_file = os.path.join(self.out_path, "result_record.txt")
        self.best_model_path = os.path.join(self.out_path, "best_global_model.pth")
        self.last_model_path = os.path.join(self.out_path, "last_global_model.pth")

        self.loss_hist = []
        self.acc_hist = []
        self.max_acc = 0

        self.run_name = f"{self.nll_name}-{alg_name}-{self.exp_name}"
        self.wb_run = wandb.init(
            config=self.args, project=self.args.proj_name, name=self.run_name
        )

    def main(self):
        # check existence of record file
        if os.path.exists(self.record_file):
            accs, _, _ = result_parser(self.record_file)
            if len(accs) >= self.args.com_round:
                self.handler._LOGGER.info(
                    f"Experiment done! Result saved in {self.record_file}!"
                )
                return

        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package

            # client side
            self.trainer.local_process(broadcast, sampled_clients, self.handler.round)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            # evaluate
            self.evaluate()

            # justify noisy clients after warmup stage:
            if self.handler.round == self.args.fednoro_warmup:
                clean_clients, noisy_clients = self.handler.justify_noisy_client()
                self.trainer.clean_clients = clean_clients
                self.trainer.noisy_clients = noisy_clients

        if self.save_last:
            torch.save(
                {
                    "model": self.handler._model.state_dict(),
                    "rounds": self.args.com_round,
                },
                self.last_model_path,
            )

        self.wb_run.log({"final_test_acc": np.mean(self.acc_hist[-10:])})
        self.wb_run.finish()
