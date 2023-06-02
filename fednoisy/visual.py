"""Visualization tools
"""
import torchvision

import seaborn as sns
from prettytable import PrettyTable
import json
from collections import Counter
import yaml
import random
import numpy as np
from matplotlib import pyplot as plt
import os
from typing import List, Dict, Tuple, Optional, Any
import numpy.typing as npt


########## For centrNLL visualization ##########
def cal_transition_matrix(
    labels: List[int], noisy_labels: List[int], num_classes: int = 10
) -> npt.NDArray[Any]:
    """Calculate transition matrix based on ground truth labels ``labels`` and noised labels ``noisy_labels``.

    Args:
        labels (List[int]): Ground truth labels.
        noisy_labels (List[int]): Labels with noise.
        num_classes: Total number of classes in the dataset.

    Return:
        np.ndarray with shape ``num_classes x num_classes``, where each entry indicates the transition probability for the labels.
    """
    cnt_matrix = np.zeros((num_classes, num_classes), int)
    labels = np.array(labels)
    noisy_labels = np.array(noisy_labels)
    for class_idx in range(num_classes):
        idxs = np.argwhere(labels == class_idx)[:, 0]
        cnt = Counter(noisy_labels[idxs])
        cnt_values = np.array([cnt[class_idx] for class_idx in range(num_classes)])
        cnt_matrix[class_idx, :] = cnt_values

    trans_matrix = cnt_matrix / cnt_matrix.sum(axis=1, keepdims=True)
    return trans_matrix


def heatmap_trans_mat(
    trans_mat: np.ndarray,
    class_to_idx: Dict[str, int],
    title: Optional[str] = None,
    figsize: Tuple[float] = (8, 6),
    save: Optional[bool] = True,
    save_path: Optional[str] = None,
) -> None:
    """Generate heatmap figure given transition matrix.

    Args:
        trans_mat (np.ndarray): transition matrix
        class_to_idx : {class_name: class_idx, ...}
        title: Title for the heatmap
        figsize: Figure size of heatmap.
        save: Whether to save heatmap figure
        save_path: Path to save the figure.
    """
    sns.set(rc={'figure.figsize': figsize})
    class_names = [x[0] for x in sorted(class_to_idx.items(), key=lambda x: x[1])]
    ax = sns.heatmap(
        trans_mat,
        cmap="crest",
        annot=True,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set(xlabel="Noisy Label", ylabel="Label")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.xaxis.tick_top()
    if title is not None:
        ax.set_title(title, fontsize=15)
    if save:
        plt.savefig(save_path, bbox_inches='tight')


def table_trans_mat(
    trans_mat, class_to_idx: Dict[str, int], title: Optional[str] = None
):
    class_names = [x[0] for x in sorted(class_to_idx.items(), key=lambda x: x[1])]
    num_classes = len(class_names)
    table = PrettyTable()
    if title is not None:
        table.title = title

    table.field_names = ["", *class_names]
    for row in range(num_classes):
        table.add_row([class_names[row], *trans_mat[row, :].tolist()])
        table.align[class_names[row]] = 'l'
    print(table)
    return table


def make_trans_table(labels, noisy_labels, num_classes, class_to_idx, title):
    """Print out table of transition matrix, return :class:`prettytable.PrettyTable`."""
    trans_matrix = cal_transition_matrix(labels, noisy_labels, num_classes=num_classes)
    table = table_trans_mat(trans_matrix, class_to_idx, title=title)
    return table


def make_heatmap(
    labels,
    noisy_labels,
    num_classes,
    class_to_idx,
    title,
    figsize: Tuple[float] = (8, 6),
    save: bool = True,
    save_path=None,
):
    trans_matrix = cal_transition_matrix(labels, noisy_labels, num_classes=num_classes)
    heatmap_trans_mat(
        trans_matrix,
        class_to_idx,
        title=title,
        figsize=figsize,
        save=save,
        save_path=save_path,
    )


if __name__ == '__main__':
    tmp_checkpoint_dir = '../tmp_checkpoint'
    noise_mode = 'sym'
    noise_ratio = 0.1
    seed = 1
    data_task_dir = f"centrNLL/cifar10_noise_mode_{noise_mode}_noise_ratio_{noise_ratio:.2f}_seed_{seed}"
    data_task_path = os.path.join(tmp_checkpoint_dir, data_task_dir)

    # === load labels and noisy labels
    trainset = torchvision.datasets.CIFAR10(
        root='/Users/liangsiqi/Documents/rawdata/cifar10/', train=True
    )
    labels = trainset.targets
    class_to_idx = trainset.class_to_idx
    noise_file_path = os.path.join(
        data_task_path,
        f"centralized_cifar10_{noise_mode}_{noise_ratio:.2f}_seed_{seed}_noise.json",
    )
    with open(noise_file_path, 'r') as nfile:
        contents = json.load(nfile)
        noisy_labels = contents['noisy_labels']

    # === calculate transition matrix
    trans_matrix = cal_transition_matrix(labels, noisy_labels, num_classes=10)
    # === generate heatmap and save it
    heatmap_trans_mat(
        trans_matrix,
        class_to_idx,
        title=f'CIFAR10-{noise_mode}_{noise_ratio:.2f} Noisy Label Transition Matrix',
        save=True,
        save_path=os.path.join(
            data_task_path, f'cifar10_{noise_mode}_{noise_ratio:.2}_trans_mat.pdf'
        ),
    )
    # === generate table version of transition matrix
    table = table_trans_mat(
        trans_matrix,
        class_to_idx,
        title=f'CIFAR10-{noise_mode}_{noise_ratio:.2f} Noisy Label Transition Matrix',
    )
    # save to html
    trans_mat_html_path = os.path.join(
        data_task_path, f'cifar10_{noise_mode}_{noise_ratio:.2}_trans_mat.html'
    )
    with open(trans_mat_html_path, 'w') as html_file:
        html_file.write(table.get_html_string())
