import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import sys
import os
from copy import deepcopy
from typing import Dict, Tuple, List, Optional

from fednoisy.utils.misc import AverageMeter

##############################################################################


##################### Loss tracking and noise modeling #######################
def track_training_loss(model, train_loader, device):
    model.eval()

    all_losses = torch.Tensor()
    all_predictions = torch.Tensor()
    all_probs = torch.Tensor()
    all_argmaxXentropy = torch.Tensor()

    for batch_idx, (imgs, labels, noisy_labels) in enumerate(train_loader):
        imgs, noisy_labels = imgs.to(device), noisy_labels.to(device)
        prediction = model(imgs)

        prediction = F.log_softmax(prediction, dim=1)
        idx_loss = F.nll_loss(prediction, noisy_labels, reduction="none")
        idx_loss.detach_()
        all_losses = torch.cat((all_losses, idx_loss.cpu()))
        probs = prediction.clone()
        probs.detach_()
        all_probs = torch.cat((all_probs, probs.cpu()))
        arg_entr = torch.max(prediction, dim=1)[1]
        arg_entr = F.nll_loss(prediction.float(), arg_entr.to(device), reduction="none")
        arg_entr.detach_()
        all_argmaxXentropy = torch.cat((all_argmaxXentropy, arg_entr.cpu()))

    loss_tr = all_losses.data.numpy()

    # outliers detection
    max_perc = np.percentile(loss_tr, 95)
    min_perc = np.percentile(loss_tr, 5)
    loss_tr = loss_tr[(loss_tr <= max_perc) & (loss_tr >= min_perc)]

    bmm_model_maxLoss = torch.FloatTensor([max_perc]).to(device)
    bmm_model_minLoss = torch.FloatTensor([min_perc]).to(device) + 10e-6

    # loss_tr = (loss_tr - bmm_model_minLoss.data.cpu().numpy()) / (bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6)
    loss_tr = (loss_tr - (min_perc + 10e-6)) / (max_perc - min_perc)

    loss_tr[loss_tr >= 1] = 1 - 10e-4
    loss_tr[loss_tr <= 0] = 10e-4

    bmm_model = BetaMixture1D(max_iters=10)
    bmm_model.fit(loss_tr)

    bmm_model.create_lookup(1)

    return bmm_model, bmm_model_maxLoss, bmm_model_minLoss


##############################################################################


############################# Mixup original #################################
def mixup_data(x, y, alpha=1.0, device=None):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.nll_loss(pred, y_a) + (1 - lam) * F.nll_loss(pred, y_b)


def train_mixUp(model, train_loader, optimizer, alpha, device):
    """NORMAL mixup

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_
        optimizer (_type_): _description_
        alpha (_type_): Alpha parameter for mixup
        device (str): _description_

    Returns:
        (float, float): train per-sample loss, train accuracy
    """
    model.train()
    acc_ = AverageMeter()
    loss_ = AverageMeter()
    for batch_idx, (imgs, labels, noisy_labels) in enumerate(train_loader):
        batch_size = len(noisy_labels)
        imgs, noisy_labels = imgs.to(device), noisy_labels.to(device)
        optimizer.zero_grad()

        inputs, targets_a, targets_b, lam = mixup_data(
            imgs, noisy_labels, alpha, device
        )

        output = model(inputs)
        output = F.log_softmax(output, dim=1)
        loss = mixup_criterion(output, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        loss_.update(loss.item(), batch_size)
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(noisy_labels.view_as(pred)).sum().item()
        acc_.update(correct / batch_size, batch_size)

    return loss_.avg, acc_.avg


##############################################################################


########################## Mixup + Dynamic Hard Bootstrapping ##################################
# Mixup with hard bootstrapping using the beta model
def reg_loss_class(mean_tab, num_classes=10):
    loss = 0
    for items in mean_tab:
        loss += (1.0 / num_classes) * torch.log((1.0 / num_classes) / items)
    return loss


def mixup_data_Boot(x, y, alpha=1.0, device=None):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def train_mixUp_HardBootBeta(
    model,
    train_loader,
    optimizer,
    bmm_model,
    bmm_model_maxLoss,
    bmm_model_minLoss,
    alpha,
    reg_term,
    num_classes,
    device,
):
    """HARD BETA bootstrapping and NORMAL mixup"""
    model.train()
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    for batch_idx, (imgs, labels, noisy_labels) in enumerate(train_loader):
        batch_size = len(noisy_labels)
        imgs, noisy_labels = imgs.to(device), noisy_labels.to(device)
        optimizer.zero_grad()

        output_x1 = model(imgs)
        output_x1.detach_()
        optimizer.zero_grad()

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(
            imgs, noisy_labels, alpha, device
        )
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        tab_mean_class = torch.mean(output_mean, -2)
        output = F.log_softmax(output, dim=1)

        B = compute_probabilities_batch(
            imgs, noisy_labels, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss
        )
        B = B.to(device)
        B[B <= 1e-4] = 1e-4
        B[B >= 1 - 1e-4] = 1 - 1e-4

        output_x1 = F.log_softmax(output_x1, dim=1)
        output_x2 = output_x1[index, :]
        B2 = B[index]

        z1 = torch.max(output_x1, dim=1)[1]
        z2 = torch.max(output_x2, dim=1)[1]

        loss_x1_vec = (1 - B) * F.nll_loss(output, targets_1, reduction="none")
        loss_x1 = torch.sum(loss_x1_vec) / len(loss_x1_vec)

        loss_x1_pred_vec = B * F.nll_loss(output, z1, reduction="none")
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)

        loss_x2_vec = (1 - B2) * F.nll_loss(output, targets_2, reduction="none")
        loss_x2 = torch.sum(loss_x2_vec) / len(loss_x2_vec)

        loss_x2_pred_vec = B2 * F.nll_loss(output, z2, reduction="none")
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)

        loss = lam * (loss_x1 + loss_x1_pred) + (1 - lam) * (loss_x2 + loss_x2_pred)

        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term * loss_reg

        loss.backward()

        optimizer.step()

        loss_.update(loss.item(), batch_size)
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(noisy_labels.view_as(pred)).sum().item()
        acc_.update(correct / batch_size, batch_size)

    return loss_.avg, acc_.avg


##############################################################################


##################### Mixup Beta Soft Bootstrapping ####################
# Mixup guided by our beta model with beta soft bootstrapping
def mixup_criterion_mixSoft(pred, y_a, y_b, B, lam, index, output_x1, output_x2):
    return torch.sum(
        (lam)
        * (
            (1 - B) * F.nll_loss(pred, y_a, reduction="none")
            + B * (-torch.sum(F.softmax(output_x1, dim=1) * pred, dim=1))
        )
        + (1 - lam)
        * (
            (1 - B[index]) * F.nll_loss(pred, y_b, reduction="none")
            + B[index] * (-torch.sum(F.softmax(output_x2, dim=1) * pred, dim=1))
        )
    ) / len(pred)


def train_mixUp_SoftBootBeta(
    model,
    train_loader,
    optimizer,
    bmm_model,
    bmm_model_maxLoss,
    bmm_model_minLoss,
    alpha,
    reg_term,
    first_flag=False,
    device=None,
):
    """SOFT BETA bootstrapping and NORMAL mixup"""
    acc_ = AverageMeter()
    loss_ = AverageMeter()
    model.train()
    for batch_idx, (imgs, labels, noisy_labels) in enumerate(train_loader):
        batch_size = len(noisy_labels)
        imgs, noisy_labels = imgs.to(device), noisy_labels.to(device)
        optimizer.zero_grad()

        output_x1 = model(imgs)
        output_x1.detach_()
        optimizer.zero_grad()

        if first_flag is True:
            B = 0.5 * torch.ones(batch_size).float().to(device)
        else:
            B = compute_probabilities_batch(
                imgs,
                noisy_labels,
                model,
                bmm_model,
                bmm_model_maxLoss,
                bmm_model_minLoss,
            )
            B = B.to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1 - 1e-4] = 1 - 1e-4

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(
            imgs, noisy_labels, alpha, device
        )
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        output = F.log_softmax(output, dim=1)

        output_x2 = output_x1[index, :]

        tab_mean_class = torch.mean(output_mean, -2)  # Columns mean

        loss = mixup_criterion_mixSoft(
            output, targets_1, targets_2, B, lam, index, output_x1, output_x2
        )
        loss_reg = reg_loss_class(tab_mean_class)
        loss = loss + reg_term * loss_reg

        loss.backward()
        optimizer.step()

        loss_.update(loss.item(), batch_size)
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(noisy_labels.view_as(pred)).sum().item()
        acc_.update(correct / batch_size, batch_size)

    return loss_.avg, acc_.avg


##############################################################################


################################ Dynamic Mixup ##################################
# Mixup guided by our beta model
def mixup_data_beta(x, y, B, device=None):
    """Returns mixed inputs, pairs of targets, and lambda"""
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    lam = (1 - B) + (1 - B[index])
    mixed_x = ((1 - B) / lam).unsqueeze(1).unsqueeze(2).unsqueeze(3) * x + (
        (1 - B[index]) / lam
    ).unsqueeze(1).unsqueeze(2).unsqueeze(3) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, index


def mixup_criterion_beta(pred, y_a, y_b):
    lam = np.random.beta(32, 32)
    return lam * F.nll_loss(pred, y_a) + (1 - lam) * F.nll_loss(pred, y_b)


def train_mixUp_Beta(
    model,
    train_loader,
    optimizer,
    bmm_model,
    bmm_model_maxLoss,
    bmm_model_minLoss,
    alpha,
    first_flag=False,
    device=None,
):
    """Dynamic mixup"""
    model.train()
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    for batch_idx, (imgs, labels, noisy_labels) in enumerate(train_loader):
        batch_size = len(noisy_labels)
        imgs, noisy_labels = imgs.to(device), noisy_labels.to(device)
        optimizer.zero_grad()

        if first_flag is True:
            B = 0.5 * torch.ones(batch_size).float().to(device)
        else:
            B = compute_probabilities_batch(
                imgs,
                noisy_labels,
                model,
                bmm_model,
                bmm_model_maxLoss,
                bmm_model_minLoss,
            )
            B = B.to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1 - 1e-4] = 1 - 1e-4

        inputs_mixed, targets_1, targets_2, index = mixup_data_beta(
            imgs, noisy_labels, B, device
        )
        output = model(inputs_mixed)
        output = F.log_softmax(output, dim=1)

        loss = mixup_criterion_beta(output, targets_1, targets_2)

        loss.backward()
        optimizer.step()

        loss_.update(loss.item(), batch_size)
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(noisy_labels.view_as(pred)).sum().item()
        acc_.update(correct / batch_size, batch_size)

    return loss_.avg, acc_.avg


################################################################################


################## Dynamic Mixup + soft2hard bootstraping ##################
def mixup_criterion_SoftHard(pred, y_a, y_b, B, index, output_x1, output_x2, Temp):
    return torch.sum(
        (0.5)
        * (
            (1 - B) * F.nll_loss(pred, y_a, reduction="none")
            + B * (-torch.sum(F.softmax(output_x1 / Temp, dim=1) * pred, dim=1))
        )
        + (0.5)
        * (
            (1 - B[index]) * F.nll_loss(pred, y_b, reduction="none")
            + B[index] * (-torch.sum(F.softmax(output_x2 / Temp, dim=1) * pred, dim=1))
        )
    ) / len(pred)


def train_mixUp_SoftHardBetaDouble(
    model,
    train_loader,
    optimizer,
    bmm_model,
    bmm_model_maxLoss,
    bmm_model_minLoss,
    alpha,
    reg_term,
    first_flag,
    Temp,
    num_classes,
    device=None,
):
    """Going from SOFT BETA bootstrapping to HARD BETA with linear temperature and Dynamic mixup"""
    model.train()
    loss_ = AverageMeter()
    acc_ = AverageMeter()

    for idx, (imgs, labels, noisy_labels) in enumerate(train_loader):
        batch_size = len(noisy_labels)
        imgs = imgs.to(device)
        noisy_labels = noisy_labels.to(device)

        output_x1 = model(imgs)
        output_x1.detach_()
        optimizer.zero_grad()

        if first_flag is True:
            B = 0.5 * torch.ones(batch_size).float().to(device)
        else:
            B = compute_probabilities_batch(
                imgs,
                noisy_labels,
                model,
                bmm_model,
                bmm_model_maxLoss,
                bmm_model_minLoss,
            )
            B = B.to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1 - 1e-4] = 1 - 1e-4

        inputs_mixed, targets_1, targets_2, index = mixup_data_beta(
            imgs, noisy_labels, B, device
        )
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        output = F.log_softmax(output, dim=1)

        output_x2 = output_x1[index, :]
        tab_mean_class = torch.mean(output_mean, -2)

        loss = mixup_criterion_SoftHard(
            output, targets_1, targets_2, B, index, output_x1, output_x2, Temp
        )
        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term * loss_reg

        loss.backward()
        optimizer.step()

        loss_.update(loss.item(), batch_size)
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(noisy_labels.view_as(pred)).sum().item()
        acc_.update(correct / batch_size, batch_size)

    return loss_.avg, acc_.avg


def compute_probabilities_batch(
    data, target, cnn_model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss
):
    cnn_model.eval()
    outputs = cnn_model(data)
    outputs = F.log_softmax(outputs, dim=1)
    batch_losses = F.nll_loss(outputs.float(), target, reduction="none")
    batch_losses.detach_()
    outputs.detach_()
    cnn_model.train()
    batch_losses = (batch_losses - bmm_model_minLoss) / (
        bmm_model_maxLoss - bmm_model_minLoss + 1e-6
    )
    batch_losses[batch_losses >= 1] = 1 - 10e-4
    batch_losses[batch_losses <= 0] = 10e-4

    # B = bmm_model.posterior(batch_losses,1)
    B = bmm_model.look_lookup(batch_losses, bmm_model_maxLoss, bmm_model_minLoss)

    return torch.FloatTensor(B)


################### CODE FOR THE BETA MODEL  ########################
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar) ** 2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(
        self,
        max_iters=10,
        alphas_init=[1, 2],
        betas_init=[2, 1],
        weights_init=[0.5, 0.5],
    ):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0 + self.eps_nan, 1 - self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t) :] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l  # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label="negative")
        plt.plot(x, self.weighted_likelihood(x, 1), label="positive")
        plt.plot(x, self.probability(x), lw=2, label="mixture")

    def __str__(self):
        return "BetaMixture1D(w={}, a={}, b={})".format(
            self.weight, self.alphas, self.betas
        )
