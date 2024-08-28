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
import pandas as pd
import os
from typing import List, Dict, Tuple, Optional, Any
import numpy.typing as npt

from fednoisy.data.NLLData.functional import partition_report


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
    sns.set(rc={"figure.figsize": figsize})
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
        plt.savefig(save_path, bbox_inches="tight")


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
        table.align[class_names[row]] = "l"
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


########## For fedNLL visualization ##########
def feddata_scatterplot(
    targets,
    client_dict,
    num_clients,
    num_classes,
    figsize=(6, 4),
    max_size=200,
    title=None,
):
    """Visualize the data distribution for each client and class in federated setting.

    Args:
        targets (_type_): List of labels, with each entry as integer number.
        client_dict (_type_): Dictionary contains sample index list for each client, ``{ client_id: indices}``
        num_clients (_type_): Number of total clients
        num_classes (_type_): Number of total classes
        figsize (tuple, optional): Figure size for scatter plot. Defaults to (6, 4).
        max_size (int, optional): Max scatter marker size. Defaults to 200.
        title (str, optional): Title for scatter plot. Defaults to None.

    Returns:
        Figure: matplotlib figure object

    Examples:
        First generate data partition:

        >>> num_clients, num_classes = 5, 10
        >>> partition = 'noniid-labeldir'
        >>> # generate federated data partition and save to local files
        >>> nll_cifar10 = FedNLLCIFAR10(globalize=True,
        ...                             partition=partition,
        ...                             num_clients=num_clients,
        ...                             dir_alpha=0.6,
        ...                             noise_mode='clean',
        ...                             noise_ratio=0.0,
        ...                             root_dir='../rawdata/cifar10/',
        ...                             out_dir='../fedNLLdata/cifar10',
        ...                             personalize=True)
        >>> nll_cifar10.create_nll_scene(seed=1)
        >>> nll_cifar10.save_nll_scene()
        >>> title= 'Train Data Distribution over Clients for Each Class'
        >>> train_fig = feddata_scatterplot(nll_cifar10.train_labels,
        ...                                 nll_cifar10.client_dict,
        ...                                 num_clients,
        ...                                 num_classes,
        ...                                 figsize=(6, 4),
        ...                                 max_size=200,
        ...                                 title=title)
        >>> plt.show(train_fig)  # Show the plot
        >>> train_fig.savefig(f'imgs/train_vis-{partition}.png')  # Save the plot
    """
    palette = sns.color_palette("Set2", num_classes)
    report_df = partition_report(
        targets, client_dict, class_num=num_classes, verbose=True
    )
    sample_stats = report_df.values[:, 1 : 1 + num_classes]
    min_max_ratio = np.min(sample_stats) / np.max(sample_stats)
    data_tuples = []
    for cid in range(num_clients):
        for k in range(num_classes):
            data_tuples.append((cid, k, sample_stats[cid, k] / np.max(sample_stats)))

    df = pd.DataFrame(data_tuples, columns=["Client", "Class", "Samples"])
    plt.figure(figsize=figsize)
    scatter = sns.scatterplot(
        data=df,
        x="Client",
        y="Class",
        size="Samples",
        hue="Class",
        palette=palette,
        legend=False,
        sizes=(max_size * min_max_ratio, max_size),
    )

    # Customize the axes and layout
    plt.xticks(range(num_clients), [f"Client {cid+1}" for cid in range(num_clients)])
    plt.yticks(range(num_classes), [f"Class {k+1}" for k in range(num_classes)])
    plt.xlabel("Clients")
    plt.ylabel("Classes")
    plt.title(title)
    return plt.gcf()


if __name__ == "__main__":
    tmp_checkpoint_dir = "../tmp_checkpoint"
    noise_mode = "sym"
    noise_ratio = 0.1
    seed = 1
    data_task_dir = f"centrNLL/cifar10_noise_mode_{noise_mode}_noise_ratio_{noise_ratio:.2f}_seed_{seed}"
    data_task_path = os.path.join(tmp_checkpoint_dir, data_task_dir)

    # === load labels and noisy labels
    trainset = torchvision.datasets.CIFAR10(
        root="/Users/liangsiqi/Documents/rawdata/cifar10/", train=True
    )
    labels = trainset.targets
    class_to_idx = trainset.class_to_idx
    noise_file_path = os.path.join(
        data_task_path,
        f"centralized_cifar10_{noise_mode}_{noise_ratio:.2f}_seed_{seed}_noise.json",
    )
    with open(noise_file_path, "r") as nfile:
        contents = json.load(nfile)
        noisy_labels = contents["noisy_labels"]

    # === calculate transition matrix
    trans_matrix = cal_transition_matrix(labels, noisy_labels, num_classes=10)
    # === generate heatmap and save it
    heatmap_trans_mat(
        trans_matrix,
        class_to_idx,
        title=f"CIFAR10-{noise_mode}_{noise_ratio:.2f} Noisy Label Transition Matrix",
        save=True,
        save_path=os.path.join(
            data_task_path, f"cifar10_{noise_mode}_{noise_ratio:.2}_trans_mat.pdf"
        ),
    )
    # === generate table version of transition matrix
    table = table_trans_mat(
        trans_matrix,
        class_to_idx,
        title=f"CIFAR10-{noise_mode}_{noise_ratio:.2f} Noisy Label Transition Matrix",
    )
    # save to html
    trans_mat_html_path = os.path.join(
        data_task_path, f"cifar10_{noise_mode}_{noise_ratio:.2}_trans_mat.html"
    )
    with open(trans_mat_html_path, "w") as html_file:
        html_file.write(table.get_html_string())
