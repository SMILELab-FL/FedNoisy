import numpy as np
import random
import os
import json
import ast
import torch
import torchnet


class AverageMeter(object):
    """Compute and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed: int = 0):
    """
    Args:
        seed (int): random seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(
    model,
    criterion,
    test_loader,
    device,
    multimodel=False,
    regression=False,
    imagenet=False,
):
    """Evaluate classify task model accuracy, allow ``model`` contains multiple networks .

    Returns:
        (loss.avg, acc.avg)
    """
    if multimodel is False:
        model.eval()
    else:
        for net in model.models:
            net.eval()

    loss_ = AverageMeter()
    acc_ = AverageMeter()

    if imagenet:
        acc5_meter = torchnet.meter.ClassErrorMeter(topk=[5], accuracy=True)
        acc5_meter.reset()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = len(labels)

            outputs = model(inputs)
            # print(f"outputs.shape={outputs.shape}, labels.shape={labels.shape}")
            if multimodel is True:
                # sum over outputs of all nets
                outputs = torch.sum(torch.stack(outputs), dim=0)

            # print(f"criterion={criterion.__class__.__name__}")
            loss = criterion(outputs, labels)

            if not regression:
                _, predicted = torch.max(outputs, 1)
                acc_.update(
                    torch.sum(predicted.eq(labels)).item() / batch_size, batch_size
                )
                if imagenet:
                    acc5_meter.add(outputs, labels)

            loss_.update(loss.item(), batch_size)

    if imagenet:
        return loss_.avg, acc_.avg, acc5_meter.value()[0] / 100

    return loss_.avg, acc_.avg


def save_json(file_name, root_dir, content):
    file_path = os.path.join(root_dir, file_name)
    with open(file_path, "w") as out_f:
        json.dump(content, out_f)
    return True


def make_dirs(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.mkdir(dir_path)
        except FileNotFoundError:
            os.makedirs(dir_path)


def make_alg_name(args):
    if args.criterion != "ce":
        alg_name = "FedAvg-RobustLoss"
    else:
        if args.mixup is True:
            alg_name = "FedAvg-Mixup"
        elif args.coteaching is True:
            alg_name = "FedAvg-Coteaching"
        elif args.dynboot is True:
            alg_name = "FedAvg-DynamicBootstrapping"
        elif args.fednoro is True:
            alg_name = "FedNoRo"
        else:
            alg_name = "FedAvg"

    return alg_name


def make_exp_name(fed_alg_name="fedavg", args=None) -> str:
    """Make logging name for federated algrithms.

    Args:
        fed_alg_name (str, optional): _description_. Defaults to "fedavg".
        args (_type_, optional): _description_. Defaults to None.

    Returns:
        str: _description_
    """
    if fed_alg_name == "fedavg":
        noisy_alg_name = None
        arch_name = f"arch={args.model}"
        opt_name = f"lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        criterion_name = make_criterion_name(args)
        if args.mixup is True:
            noisy_alg_name = f"mixup=True-mixup_alpha={args.mixup_alpha:.2f}"
        elif args.coteaching is True:
            noisy_alg_name = f"coteaching=True-coteaching_forget_rate={args.coteaching_forget_rate}-coteaching_num_gradual={args.coteaching_num_gradual}-coteaching_exponent={args.coteaching_exponent}"
        elif args.dynboot is True:
            noisy_alg_name = f"dynboot=True-dynboot_mixup={args.dynboot_mixup}-dynboot_alpha={args.dynboot_alpha:.2f}-dynboot_bootbeta={args.dynboot_bootbeta}-dynboot_reg={args.dynboot_reg:.2f}"

    elif fed_alg_name == "fednoro":
        arch_name = f"arch={args.model}"
        opt_name = f"-opt={args.fednoro_opt}-lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        criterion_name = make_criterion_name(args)
        noisy_alg_name = f"warmup={args.fednoro_warmup}-begin={args.fednoro_begin}-end={args.fednoro_end}-a={args.fednoro_a:.2f}"

    other_name = f"com_round={args.com_round}-local_epochs={args.epochs}-sample_ratio={args.sample_ratio:.2f}-batch_size={args.batch_size}-seed={args.seed}"

    if noisy_alg_name is None:
        exp_name = "-".join(
            [fed_alg_name, criterion_name, arch_name, opt_name, other_name]
        )
    else:
        exp_name = "-".join(
            [
                fed_alg_name,
                criterion_name,
                noisy_alg_name,
                arch_name,
                opt_name,
                other_name,
            ]
        )

    return exp_name


def make_exp_name_centr(alg_name="dividemix", args=None):
    if alg_name == "crossentropy":
        noise_name = f"noise_mode={args.noise_mode}-noise_ratio={args.noise_ratio:.2f}"
        opt_name = f"lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        other_name = f"num_epochs={args.num_epochs}-batch_size={args.batch_size}-seed={args.seed}"
        exp_name = "-".join([noise_name, opt_name, other_name])

    elif alg_name == "dividemix":
        noise_name = f"noise_mode={args.noise_mode}-noise_ratio={args.noise_ratio:.2f}"
        alg_param = f"p_threshold={args.p_threshold:.2f}-lambda_u={args.lambda_u}-T={args.T:.2f}-alpha={args.alpha:.2f}"
        opt_name = f"lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        other_name = f"num_epochs={args.num_epochs}-batch_size={args.batch_size}-seed={args.seed}"
        exp_name = "-".join([noise_name, alg_name, alg_param, opt_name, other_name])

    elif alg_name == "coteaching":
        pass
    return exp_name


def make_criterion_name(args):
    criterion_name = f"criterion={args.criterion}"

    if args.criterion == "ce":
        criterion_param = ""
    elif args.criterion == "sce":
        criterion_param = f"sce_alpha={args.sce_alpha:.2f}-sce_beta={args.sce_beta:.2f}"
    elif args.criterion in ["rce", "nce", "nrce"]:
        criterion_param = f"loss_scale={args.loss_scale:.2f}"
    elif args.criterion == "gce":
        criterion_param = f"gce_q={args.gce_q:.2f}"
    elif args.criterion == "ngce":
        criterion_param = f"loss_scale={args.loss_scale:.2f}-gce_q={args.gce_q:.2f}"
    elif args.criterion in ["mae", "nmae"]:
        criterion_param = f"loss_scale={args.loss_scale:.2f}"
    elif args.criterion in ["focal", "nfocal"]:
        if args.focal_alpha is None:
            criterion_param = f"focal_gamma={args.focal_gamma:.2f}-focal_alpha=None"
        else:
            criterion_param = (
                f"focal_gamma={args.focal_gamma:.2f}-focal_alpha={args.focal_alpha:.2f}"
            )
    criterion_name = "-".join([criterion_name, criterion_param])
    return criterion_name


def serialize_model(model: torch.nn.Module) -> torch.Tensor:
    parameters = [param.data.view(-1) for param in model.state_dict().values()]
    m_parameters = torch.cat(parameters)
    m_parameters = m_parameters.cpu()

    return m_parameters


def deserialize_model(
    model: torch.nn.Module, serialized_parameters: torch.Tensor, mode="copy"
):
    current_index = 0  # keep track of where to read from grad_update

    for param in model.state_dict().values():
        numel = param.numel()
        size = param.size()
        if mode == "copy":
            param.copy_(
                serialized_parameters[current_index : current_index + numel].view(size)
            )
        elif mode == "add":
            param.add_(
                serialized_parameters[current_index : current_index + numel].view(size)
            )
        else:
            raise ValueError(
                'Invalid deserialize mode {}, require "copy" or "add" '.format(mode)
            )
        current_index += numel


def result_parser(result_path):
    """_summary_

    Args:
        result_path (str): _description_

    Returns:
        tuple[List[float], List[float], Dict]: _description_
    """
    with open(result_path, "r") as f:
        lines = f.readlines()
    # hist accuracy
    accs = [float(item) for item in lines[1].strip()[5:-1].split(", ")]
    # hist losses
    losses = [float(item) for item in lines[2].strip()[6:-1].split(", ")]
    # hyperparameter setting
    setting_dict = ast.literal_eval(lines[0].strip())
    return accs, losses, setting_dict
