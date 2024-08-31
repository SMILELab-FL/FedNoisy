import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy

from fednoisy.utils import misc as misc
from fedlab.utils import Logger, Aggregators


def model_dist(w1, w2):
    return torch.norm(w1 - w2).cpu().item()


class LogitAdjust(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)


class LA_KD(nn.Module):
    def __init__(self, cls_num_list, tau=1):
        super(LA_KD, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

    def forward(self, x, target, soft_target, w_kd):
        x_m = x + self.m_list
        log_pred = torch.log_softmax(x_m, dim=-1)
        log_pred = torch.where(
            torch.isinf(log_pred), torch.full_like(log_pred, 0), log_pred
        )

        kl = F.kl_div(log_pred, soft_target, reduction="batchmean")

        return w_kd * kl + (1 - w_kd) * F.nll_loss(log_pred, target)


def train_LA(
    model, optimizer, train_loader, epochs, class_num_list, logger, round, cid, device
):
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        args (_type_): _description_
        class_num_list (_type_): np.array with shape (num_classes,), label counts of current client
    """
    model.train()
    epoch_loss = []
    ce_criterion = LogitAdjust(cls_num_list=class_num_list)

    for epoch in range(epochs):
        logger.info(f"Round {round} client-{cid} local train epoch [{epoch}/{epochs}]")
        batch_loss = []
        for batch_idx, (imgs, _, noisy_labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            noisy_labels = noisy_labels.to(device)

            logits = model(imgs)
            loss = ce_criterion(logits, noisy_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        avg_batch_loss = np.array(batch_loss).mean()
        epoch_loss.append(avg_batch_loss)
        logger.info(
            f"Round {round} client-{cid} local train epoch [{epoch}/{epochs}] training loss: {avg_batch_loss:.4f}"
        )

    logger.info(
        f"Round {round} client-{cid} local train epoch training loss: {np.array(epoch_loss).mean():.4f}"
    )


def train_FedNoRo(
    student_model,
    teacher_model,
    train_loader,
    weight_kd,
    epochs,
    lr,
    args,
    class_num_list,
    logger,
    round,
    cid,
    device,
):
    student_model.train()
    teacher_model.eval()
    # set the optimizer
    if args.fednoro_opt == "sgd":
        optimizer = torch.optim.SGD(
            student_model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )  # default weight_decay is 1e-4

    elif args.fednoro_opt == "adam":
        optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )  # default weight_decay is 5e-4

    criterion = LA_KD(cls_num_list=class_num_list)
    epoch_loss = []
    for epoch in range(epochs):
        batch_loss = []
        for batch_idx, (imgs, _, noisy_labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            noisy_labels = noisy_labels.to(device)

            logits = student_model(imgs)

            with torch.no_grad():
                teacher_output = teacher_model(imgs)
                soft_labels = torch.softmax(teacher_output / 0.8, dim=1)

            loss = criterion(logits, noisy_labels, soft_labels, weight_kd)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

        avg_batch_loss = np.array(batch_loss).mean()
        epoch_loss.append(avg_batch_loss)
        logger.info(
            f"Round {round} client-{cid} local train epoch [{epoch}/{epochs}] training loss: {avg_batch_loss:.4f}"
        )

    logger.info(
        f"Round {round} client-{cid} local train epoch training loss: {np.array(epoch_loss).mean():.4f}"
    )
    return misc.serialize_model(student_model)


def partialDaAgg(
    parameters_list, weights, sampled_clients, clean_clients, noisy_clients
):
    param_dict, weight_dict = {}, {}
    for idx in range(len(sampled_clients)):
        param_dict[sampled_clients[idx]] = deepcopy(parameters_list[idx])
        weight_dict[sampled_clients[idx]] = deepcopy(weights[idx])

    weight_sum = sum(list(weight_dict.values()))
    for key in sampled_clients:
        weight_dict[key] /= weight_sum

    selected_noisy_clients = [cid for cid in sampled_clients if cid in noisy_clients]
    selected_clean_clients = [cid for cid in sampled_clients if cid in clean_clients]

    distance = {cid: 0 for cid in param_dict}
    for n_cid in selected_noisy_clients:
        dist = []
        for c_cid in selected_clean_clients:
            dist.append(model_dist(param_dict[n_cid], param_dict[c_cid]))
        distance[n_cid] = min(dist)

    dist_max = max(list(distance.values()))
    if dist_max == 0.0:
        dist_max = 1.0
    # print(f"dist_max: {dist_max:.10f}")

    updated_weights = [
        weight_dict[cid] * np.exp(-distance[cid] / dist_max) for cid in sampled_clients
    ]
    return Aggregators.fedavg_aggregate(parameters_list, updated_weights)


def get_output(model, loader, softmax=False, criterion=None, device=None):
    model.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, _, noisy_labels) in enumerate(loader):
            images = images.to(device)
            noisy_labels = noisy_labels.to(device)
            noisy_labels = noisy_labels.long()
            if softmax == True:
                outputs = model(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, noisy_labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                if criterion is not None:
                    loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                if criterion is not None:
                    loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def sigmoid_rampup(current, begin, end):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = np.clip(current, begin, end)
    phase = 1.0 - (current - begin) / (end - begin)
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(rnd, begin, end):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(rnd, begin, end)
