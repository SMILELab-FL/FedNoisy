from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from torchnet.meter import AUCMeter

from copy import deepcopy
import random
import numpy as np
from typing import Dict, List
from PIL import Image
import json
import os

from . import CLASS_NUM
from . import TRAIN_SAMPLE_NUM
from . import CIFAR10_TRANSITION_MATRIX
from . import NORM_VALUES


class DivideMixCIFAR10(Dataset):
    trainset_filename = 'cifar10-trainset.pt'
    testset_filename = 'cifar10-testset.pt'

    def __init__(self, root_dir, transform, mode, noise_file='', pred=[],
                 probability=[]):
        self.transform = transform
        self.mode = mode
        self.root_dir = root_dir
        self.noise_file = noise_file

        if self.mode == 'test':
            # load test set from local file
            file_path = os.path.join(self.root_dir, self.testset_filename)
            entry = torch.load(file_path)
            self.test_data = deepcopy(entry['data'])
            self.test_labels = deepcopy(entry['labels'])
        else:
            # if mode is 'labeled', 'unlabeled' or 'all'
            # load train set from local file
            file_path = os.path.join(self.root_dir, self.trainset_filename)
            entry = torch.load(file_path)
            train_data = entry['data']
            train_labels = entry['labels']

            # load noisy labels from local file
            file_path = os.path.join(self.root_dir, self.noise_file)  # json file
            with open(file_path, 'r') as f:
                entry = json.load(f)
                noisy_labels = entry['noisy_labels']

            if self.mode == 'all':
                self.train_data = deepcopy(train_data)
                self.noisy_labels = deepcopy(noisy_labels)
            else:
                if self.mode == 'labeled':
                    pred_idxs = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idxs]

                    clean = (np.array(noisy_labels) == np.array(train_labels))
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability, clean)
                    auc, _, _ = auc_meter.value()
                    self.auc = auc.tolist()
                    # log.write('Number of labeled samples:%d   AUC:%.3f\n' % (pred.sum(), auc))
                    # log.flush()

                elif self.mode == 'unlabeled':
                    pred_idxs = (1 - pred).nonzero()[0]

                self.train_data = deepcopy(train_data[pred_idxs])
                self.noisy_labels = deepcopy([noisy_labels[i] for i in pred_idxs])
                print("%s data has a size of %d" % (self.mode, len(self.noisy_labels)))

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noisy_labels[index], self.probability[
                index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob

        elif self.mode == "unlabeled":
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2

        elif self.mode == "all":
            img, target = self.train_data[index], self.noisy_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index

        elif self.mode == "test":
            img, target = self.test_data[index], self.test_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class DivideMixCIFAR100(DivideMixCIFAR10):
    trainset_filename = 'cifar100-trainset.pt'
    testset_filename = 'cifar100-testset.pt'


class DivideMixClothing1M(Dataset):
    dataset_name = 'clothing1m'
    num_classes = CLASS_NUM['clothing1m']
    trainset_filename = 'clothing1m-trainset.pt'
    testset_filename = 'clothing1m-testset.pt'
    valset_filename = 'clothing1m-valset.pt'

    def __init__(self, root_dir, transform, mode, num_samples=0, pred=[],
                 probability=[], paths=[]):
        self.transform = transform
        self.mode = mode
        self.root_dir = root_dir

        trainset_path = os.path.join(self.root_dir, self.trainset_filename)
        testset_path = os.path.join(self.root_dir, self.testset_filename)
        valset_path = os.path.join(self.root_dir, self.valset_filename)

        if self.mode in ['labeled', 'all']:
            train_entry = torch.load(trainset_path)
            self.train_labels = train_entry['train_labels']  # TODO: deepcopy?
        elif self.mode in ['test', 'val']:
            test_entry = torch.load(testset_path)
            self.test_labels = test_entry['test_labels']  # TODO: deepcopy?

        if self.mode == 'all':
            train_imgs = train_entry['train_imgs']
            random.shuffle(train_imgs)
            sample_num_each_class = torch.zeros(self.num_classes)
            self.train_imgs = []
            cur_sample_num = 0
            for impath in train_imgs:
                label = self.train_labels[impath]
                if sample_num_each_class[label] < (
                        num_samples / 14) and cur_sample_num < num_samples:
                    self.train_imgs.append(impath)
                    sample_num_each_class[label] += 1
                    cur_sample_num += 1
            random.shuffle(self.train_imgs)

        elif self.mode == 'labeled':
            train_imgs = paths
            pred_idxs = pred.nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idxs]
            self.probability = [probability[i] for i in pred_idxs]
            print(f"{self.mode} data has a size of {len(self.train_imgs)}")

        elif self.mode == 'unlabeled':
            train_imgs = paths
            pred_idxs = (1 - pred).nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idxs]
            self.probability = [probability[i] for i in pred_idxs]
            print(f"{self.mode} data has a size of {len(self.train_imgs)}")

        elif self.mode == 'test':
            self.test_imgs = test_entry['test_imgs']

        elif self.mode == 'val':
            entry = torch.load(valset_path)
            self.val_imgs = entry['val_imgs']

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob

        elif self.mode == 'unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2

        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, img_path

        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        elif self.mode == 'val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)


class DivideMixWebVision(Dataset):
    dataset_name = 'webvision'
    trainset_filename = 'webvision-trainset.pt'
    testset_filename = 'webvision-testset.pt'
    valset_filename = 'webvision-valset.pt'

    def __init__(self, root_dir, transform, mode, num_classes=50, pred=[], probability=[]):
        self.transform = transform
        self.mode = mode
        self.root_dir = root_dir
        self.num_classes = num_classes

        trainset_path = os.path.join(self.root_dir, self.trainset_filename)
        testset_path = os.path.join(self.root_dir, self.testset_filename)
        valset_path = os.path.join(self.root_dir, self.valset_filename)

        if self.mode == 'test':
            entry = torch.load(valset_path)
            self.val_imgs = entry['val_imgs']
            self.val_labels = entry['val_labels']
        else:
            entry = torch.load(trainset_path)
            train_imgs = entry['train_imgs']
            train_labels = entry['train_labels']
            self.train_labels = train_labels
            if self.mode == 'all':
                self.train_imgs = train_imgs
            else:
                if self.mode == 'labeled':
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]
                    self.probability = [probability[i] for i in pred_idx]
                    print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))
                    labeled_ratio = pred.sum() / len(self.train_imgs)
                    self.labeled_ratio = labeled_ratio
                elif self.mode == 'unlabeled':
                    pred_idx = (1 - pred).nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]
                    print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob

        elif self.mode == 'unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2

        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index

        elif self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)


class DivideMixImageNet(Dataset):
    dataset_name = 'imagenet'
    valset_filename = 'webvision-valset.pt'

    def __init__(self, root_dir, transform, num_classes=50):
        self.transform = transform
        self.root_dir = root_dir
        self.num_classes = num_classes
        valset_path = os.path.join(self.root_dir, self.valset_filename)
        entry = torch.load(valset_path)
        self.val_data = entry['imagenet_val_data']

    def __getitem__(self, index):
        img_path, target = self.val_data[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.val_data)


class DivideMixCIFAR10Loader(object):
    dataset_name = 'cifar10'

    def __init__(self, noise_ratio, noise_mode, batch_size, num_workers, root_dir,
                 noise_file=''):
        self.noise_ratio = noise_ratio
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES[self.dataset_name]),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES[self.dataset_name]),
        ])

    def run(self, mode, pred=[], prob=[]):
        if self.dataset_name == 'cifar10':
            cifar_dataset = DivideMixCIFAR10
        elif self.dataset_name == 'cifar100':
            cifar_dataset = DivideMixCIFAR100

        if mode == 'warmup':
            all_dataset = cifar_dataset(root_dir=self.root_dir,
                                        transform=self.transform_train,
                                        mode="all",
                                        noise_file=self.noise_file)
            train_loader = DataLoader(dataset=all_dataset,
                                      batch_size=self.batch_size * 2,
                                      shuffle=True,
                                      num_workers=self.num_workers)
            return train_loader

        elif mode == 'train':
            # labeled part
            labeled_dataset = cifar_dataset(root_dir=self.root_dir,
                                            transform=self.transform_train,
                                            mode='labeled',
                                            noise_file=self.noise_file,
                                            pred=pred,
                                            probability=prob)
            labeled_loader = DataLoader(dataset=labeled_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        num_workers=self.num_workers)

            # unlabeled part
            unlabeled_dataset = cifar_dataset(root_dir=self.root_dir,
                                              transform=self.transform_train,
                                              mode='unlabeled',
                                              noise_file=self.noise_file,
                                              pred=pred)
            unlabeled_loader = DataLoader(dataset=unlabeled_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=self.num_workers)
            return labeled_loader, unlabeled_loader, labeled_dataset.auc

        elif mode == 'test':
            test_dataset = cifar_dataset(root_dir=self.root_dir,
                                         transform=self.transform_test,
                                         mode="test")
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers)
            return test_loader

        elif mode == "eval_train":
            eval_dataset = cifar_dataset(root_dir=self.root_dir,
                                         transform=self.transform_test,
                                         mode='all',
                                         noise_file=self.noise_file)
            eval_loader = DataLoader(dataset=eval_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers)
            return eval_loader


class DivideMixCIFAR100Loader(DivideMixCIFAR10Loader):
    dataset_name = 'cifar100'


class DivideMixClothing1MLoader(object):
    dataset_name = 'clothing1m'

    def __init__(self, root_dir, batch_size, num_batches, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root_dir = root_dir

        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES[self.dataset_name]),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES[self.dataset_name]),
        ])

    def run(self, mode, pred=[], prob=[], paths=[]):
        if mode == 'warmup':
            warmup_dataset = DivideMixClothing1M(root_dir=self.root_dir,
                                                 transform=self.transform_train,
                                                 mode='all',
                                                 num_samples=self.num_batches * self.batch_size * 2)
            warmup_loader = DataLoader(dataset=warmup_dataset,
                                       batch_size=self.batch_size * 2,
                                       shuffle=True,
                                       num_workers=self.num_workers)
            return warmup_loader

        elif mode == 'train':
            labeled_dataset = DivideMixClothing1M(root_dir=self.root_dir,
                                                  transform=self.transform_train,
                                                  mode='labeled',
                                                  pred=pred,
                                                  probability=prob,
                                                  paths=paths)
            labeled_loader = DataLoader(dataset=labeled_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        num_workers=self.num_workers)
            unlabeled_dataset = DivideMixClothing1M(root_dir=self.root_dir,
                                                    transform=self.transform_train,
                                                    mode='unlabeled',
                                                    pred=pred,
                                                    probability=prob,
                                                    paths=paths)
            unlabeled_loader = DataLoader(dataset=unlabeled_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=self.num_workers)
            return labeled_loader, unlabeled_loader

        elif mode == 'eval_train':
            eval_dataset = DivideMixClothing1M(root_dir=self.root_dir,
                                               transform=self.transform_test,
                                               mode='all',
                                               num_samples=self.num_batches * self.batch_size)
            eval_loader = DataLoader(dataset=eval_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers)
            return eval_loader

        elif mode == 'test':
            test_dataset = DivideMixClothing1M(root_dir=self.root_dir,
                                               transform=self.transform_test,
                                               mode='test')
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=1000,
                                     shuffle=False,
                                     num_workers=self.num_workers)
            return test_loader

        elif mode == 'val':
            val_dataset = DivideMixClothing1M(root_dir=self.root_dir,
                                              transform=self.transform_test,
                                              mode='val')
            val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=1000,
                                    shuffle=False,
                                    num_workers=self.num_workers)
            return val_loader


class DivideMixWebVisionLoader(object):
    dataset_name = 'webvision'

    def __init__(self, root_dir, batch_size, num_workers, num_classes=50):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.root_dir = root_dir

        self.transform_train = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES[self.dataset_name]),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES[self.dataset_name]),
        ])
        self.transform_imagenet = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_VALUES[self.dataset_name]),
        ])

    def run(self, mode, pred=[], prob=[]):
        if mode == 'warmup':
            warmup_dataset = DivideMixWebVision(root_dir=self.root_dir,
                                                transform=self.transform_train,
                                                mode='all',
                                                num_classes=self.num_classes)
            warmup_loader = DataLoader(dataset=warmup_dataset,
                                       batch_size=self.batch_size * 2,
                                       shuffle=True,
                                       num_workers=self.num_workers,
                                       pin_memory=True)
            return warmup_loader

        elif mode == 'train':
            labeled_dataset = DivideMixWebVision(root_dir=self.root_dir,
                                                 transform=self.transform_train,
                                                 mode='labeled',
                                                 num_classes=self.num_classes,
                                                 pred=pred,
                                                 probability=prob)
            labeled_loader = DataLoader(dataset=labeled_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        num_workers=self.num_workers,
                                        pin_memory=True)

            unlabeled_dataset = DivideMixWebVision(root_dir=self.root_dir,
                                                   transform=self.transform_train,
                                                   mode='unlabeled',
                                                   num_classes=self.num_classes,
                                                   pred=pred)
            unlabeled_loader = DataLoader(dataset=unlabeled_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=self.num_workers,
                                          pin_memory=True)
            return labeled_loader, unlabeled_loader, labeled_dataset.labeled_ratio

        elif mode == 'test':
            test_dataset = DivideMixWebVision(root_dir=self.root_dir,
                                              transform=self.transform_test,
                                              mode='test',
                                              num_classes=self.num_classes)
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=self.batch_size * 20,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=True)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = DivideMixWebVision(root_dir=self.root_dir,
                                              transform=self.transform_test,
                                              mode='all',
                                              num_classes=self.num_classes)
            eval_loader = DataLoader(dataset=eval_dataset,
                                     batch_size=self.batch_size * 20,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=True)
            return eval_loader

        elif mode == 'imagenet':
            imagenet_val = DivideMixImageNet(root_dir=self.root_dir,
                                             transform=self.transform_imagenet,
                                             num_classes=self.num_classes)
            imagenet_loader = DataLoader(dataset=imagenet_val,
                                         batch_size=self.batch_size * 20,
                                         shuffle=False,
                                         num_workers=self.num_workers,
                                         pin_memory=True)
            return imagenet_loader
