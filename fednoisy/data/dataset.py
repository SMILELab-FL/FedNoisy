import torch
import argparse
import sys
import os
from typing import Dict, Tuple, List, Optional

from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


from fedlab.contrib.dataset.basic_dataset import FedDataset

from fednoisy.data.NLLData import functional as nllF


class FedNLLDataset(FedDataset):
    """Get dataset FedNLL setting from local files, can directly generate DataLoader
    used for train/test.

    Args:
        args:
        train_preload (bool): whether to preload train data into memory
        test_preload (bool): whether to preload test data into memory


    """

    def __init__(self, args, train_preload=False, test_preload=False) -> None:
        self.args = args
        nll_name = nllF.FedNLL_name(**vars(args))
        nll_filename = f"{nll_name}_seed_{args.seed}_setting.pt"
        nll_file_path = os.path.join(args.data_dir, nll_filename)
        self.nll_folder = os.path.join(args.data_dir, f"{nll_name}_seed_{args.seed}")
        self.train_preload = train_preload
        self.test_preload = test_preload
        self.personalize = args.personalize

        if train_preload:
            self.train_datasets = {cid: None for cid in range(args.num_clients)}
            for cid in range(args.num_clients):
                self.train_datasets[cid] = torch.load(
                    os.path.join(self.nll_folder, f"train-data{cid}.pkl")
                )
            print(f"Client train datasets preloaded.")
        if test_preload:
            self.test_dataset = {
                -1: torch.load(os.path.join(self.nll_folder, "test-data.pkl"))
            }  # use key -1 for global test set
            if args.dataset == "webvision":
                self.test_dataset[-2] = torch.load(
                    os.path.join(self.nll_folder, "imagenet-test-data.pkl")
                )  # use key -2 for global imagenet val set
            if self.personalize:
                # load local test set
                for cid in range(args.num_clients):
                    self.test_dataset[cid] = torch.load(
                        os.path.join(self.nll_folder, f"test-data{cid}.pkl")
                    )  # load local test set

            print(f"Test datasets preloaded.")

    def get_dataset(self, cid=None, train=True, imagenet=False):
        if train:
            if self.train_preload:
                dataset = self.train_datasets[cid]
            else:
                dataset = torch.load(
                    os.path.join(self.nll_folder, f"train-data{cid}.pkl")
                )
        else:
            if self.test_preload:
                if imagenet:
                    dataset = self.test_dataset[-2]
                else:
                    dataset = self.test_dataset[-1]
            else:
                if imagenet:
                    dataset = torch.load(
                        os.path.join(self.nll_folder, "imagenet-test-data.pkl")
                    )
                else:
                    dataset = torch.load(os.path.join(self.nll_folder, "test-data.pkl"))

        return dataset

    def get_dataloader(
        self,
        cid=-1,
        train=True,
        batch_size=64,
        num_workers=4,
        return_label_space=False,
        imagenet=False,
    ):
        # if train:
        dataset = self.get_dataset(cid, train, imagenet=imagenet)
        if train:
            shuffle = True
        else:
            shuffle = False

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        if return_label_space:
            return data_loader, dataset.label_space

        return data_loader
