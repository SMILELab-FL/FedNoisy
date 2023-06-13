from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
import torchvision

from typing import Dict, List, Set, Optional, Any
from fednoisy.data import CLASS_NUM


class NoisyDataset(Dataset):
    def __init__(
        self,
        data,
        labels,
        noisy_labels=None,
        train=True,
        transform=None,
        folder_data=False,
    ) -> None:
        """_summary_

        Args:
            data (_type_): _description_
            labels (_type_): _description_
            noisy_labels (_type_, optional): _description_. Defaults to None.
            train (bool, optional): _description_. Defaults to True.
            transform (_type_, optional): _description_. Defaults to None.
            folder_data (bool, optional): For large dataset (i.e., Clothing1M), ``folder_data`` is ``True``, that only image paths are stored in ``Dataset``; for small dataset (i.e., CIFAR-10, CIFAR-100, SVHN, MNIST), ``foldre_data`` is ``False``, that images are stored in ``Dataset``. Defaults to ``False``.
        """
        self.data = data
        self.labels = labels
        self.noisy_labels = noisy_labels
        self.train = train
        self.transform = transform
        self.folder_data = folder_data

        if self.folder_data is False:
            shape = data[0].shape
            ndim = len(shape)
            if ndim == 2:
                self.mode = "L"  # MNIST
            elif ndim == 3 and shape[-1] == 3:
                self.mode = "RGB"  # CIFAR10 & CIFAR100 & SVHN

    def __getitem__(self, index):
        if self.folder_data is False:
            # CIFAR10 & CIFAR100 & SVHN & MNIST
            img, label = self.data[index], self.labels[index]
            img = Image.fromarray(img, mode=self.mode)
        else:
            # clothing1m data
            img_path, label = self.data[index], self.labels[index]
            img = Image.open(img_path).convert("RGB")

        img = self.transform(img)

        if self.train:
            noisy_label = self.noisy_labels[index]
            return img, label, noisy_label
        else:
            return img, label

    def __len__(self):
        return len(self.labels)


def FedNLL_name(
    dataset, globalize, partition, num_clients=10, noise_mode="clean", **kw
):
    if noise_mode == "clean":
        kw["noise_ratio"] = 0.0

    prefix = "fedNLL"
    if partition == "noniid-#label":
        partition_param = f"{kw['major_classes_num']}"
    elif partition == "noniid-quantity":
        partition_param = f"{kw['dir_alpha']}"
    elif partition == "noniid-labeldir":
        partition_param = f"{kw['dir_alpha']:.2f}_{kw['min_require_size']}"
    else:
        # IID
        partition_param = ""
    partition_setting = f"{num_clients}_{partition}_{partition_param}"
    noise_setting = ""
    if noise_mode != "real":
        if globalize is False:
            noise_param = f"local_{noise_mode}_min_{kw['min_noise_ratio']:.2f}_max_{kw['max_noise_ratio']:.2f}"
        else:
            noise_param = f"global_{noise_mode}_{kw['noise_ratio']:.2f}"  # if noise_ratio is a float number
    else:
        if dataset == "clothing1m":
            noise_param = f"{noise_mode}_{kw['num_samples']}"
    setting = f"{partition_setting}_{noise_param}"
    return f"{prefix}_{dataset}_{setting}"


def check_centr_noisy_setting(noise_mode: str, noise_ratio: float):
    if noise_mode == "clean":
        if noise_ratio != 0.0:
            warnings.warn(
                f"noise_ratio should be 0.0 when noise_mode='{noise_mode}', rather than {noise_ratio:.2f}!"
            )
            noise_ratio = 0.0
    elif noise_mode in ["sym", "asym", "real"]:
        if noise_ratio > 1.0 or noise_ratio <= 0.0:
            raise ValueError(
                f"noise_ratio={noise_ratio} is not allowed for noise_mode='{noise_mode}'. "
                f"noise_ratio should be in range of (0, 1]."
            )
    else:
        raise ValueError(
            f"noise_mode='{noise_mode}' is not supported. Choose from ['sym', 'asym', 'real', 'clean']."
        )

    return noise_ratio


def symmetric_label_flipping(
    labels: List[int], class_space: Set[int], noise_ratio: float = 0.1
) -> List[int]:
    """Symetric flipping when noise_ratio >0; When noise_ratio=0.0, same as clean setting."""
    sample_num = len(labels)
    idxs = list(range(sample_num))
    random.shuffle(idxs)
    num_noise = int(sample_num * noise_ratio)
    noisy_idxs = idxs[:num_noise]  # indices of candidate noisy sample

    noisy_labels = []
    for i in range(sample_num):
        if i in noisy_idxs:
            y_ = other_class(class_space, labels[i])
            noisy_labels.append(y_)
        else:
            noisy_labels.append(labels[i])

    return noisy_labels


def build_for_cifar100(size: int, noise: float):
    """random flip between two random classes. Pairwise label noise"""
    assert (noise >= 0.0) and (noise <= 1.0)

    P = (1.0 - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i + 1] = noise

    # adjust last row
    P[size - 1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def subclasses_transmat(subclass_size: int, first_class: int) -> Dict[int, int]:
    """Make sub-transition matrix of current subclass group for CIFAR-100 asymmetric label noise.

    Args:
        subclass_size (int): Number of subclasses
        first_class (int): Index for the first class for the current subclass group.

    Returns:
        Dict[int, int]: _description_
    """
    sub_trans_mat = {}
    for i in np.arange(subclass_size - 1):
        sub_trans_mat[first_class + i] = first_class + i + 1
    sub_trans_mat[first_class + subclass_size - 1] = first_class
    return sub_trans_mat


def build_cifar100_transmat() -> Dict[int, int]:
    nb_classes = 100
    nb_superclasses = 20
    nb_subclasses = 5
    trans_mat = dict()
    for i in range(nb_superclasses):
        init, end = i * nb_subclasses, (i + 1) * nb_subclasses
        sub_trans_mat = subclasses_transmat(nb_subclasses, init)
        trans_mat.update(sub_trans_mat)
    return trans_mat


def pair_label_flipping(
    labels: List[int],
    noise_ratio: float = 0.1,
    transition: Dict[int, int] = None,
) -> List[int]:
    """Asymmetric label flipping using transition matrix directly."""
    sample_num = len(labels)
    idxs = list(range(sample_num))
    random.shuffle(idxs)
    num_noise = int(sample_num * noise_ratio)
    noisy_idxs = idxs[:num_noise]  # indices of candidate noisy sample
    noisy_labels = []

    for i in range(sample_num):
        if i in noisy_idxs:
            y_ = transition[labels[i]]
            noisy_labels.append(y_)
        else:
            noisy_labels.append(labels[i])

    return noisy_labels


def generate_noisy_labels(
    labels: List[int],
    noise_mode: str = "sym",
    noise_ratio: float = 0.1,
    transition: Optional[Dict[int, int]] = None,
    num_classes: int = 10,
) -> List[int]:
    """
    Generate noisy labels given ground truth label.

    Args:
        labels (List[int]): List of ground truth labels of dataset.
        noise_mode (str): noise mode for centralized noisy label. 'sym' for symmetric noise; 'asym' for asymmetric noise; 'clean' for clean data.
        noise_ratio (float)): Non-negative noise ratio. If ``noise_mode='clean'``, ``noise_ratio`` should be ``0``; if ``noise_mode='sym'`` or ``noise_mode='asym'``, ``noise_ratio`` should be positive.
        transition (Optional[Dict[int, int]]): Transition matrix for asymmetric noise. It maps ground truth label to designed noisy label. For example, ``{0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}``.
        num_classes (int): Number of label classes. Default as 10.

    Returns:
    """
    class_space = set(range(num_classes))
    noise_ratio = check_centr_noisy_setting(
        noise_mode, noise_ratio
    )  # check for validation of noisy setting
    if noise_mode == "asym":
        noisy_labels = pair_label_flipping(labels, noise_ratio, transition)

    else:
        # clean or symmetric
        noisy_labels = symmetric_label_flipping(
            labels, class_space, noise_ratio=noise_ratio
        )

    return noisy_labels


def generate_local_noisy_labels(
    labels: List[int],
    noise_mode: str = "sym",
    noise_ratio: float = 0.1,
    transition: Dict[int, int] = None,
    dataset: str = "ciafr10",
) -> List[int]:
    """Generate localized noisy labels for current ``labels`` based on noise setting. The label
    space will not change after adding noise, which means if there are only classes ``{1, 3, 5}`` in
    ``labels``, then corresponding noisy labels can only contain classes in ``{1, 3, 5}`` too.

    Args:
        labels:
        noise_mode:
        noise_ratio:
        transition (Dict[int, int]): Transition matrix for asymmetric noise.
        dataset (str): Dataset name.

    Returns:

    """
    class_space = set(labels)
    sample_num = len(labels)
    idxs = list(range(sample_num))
    random.shuffle(idxs)
    num_noise = int(sample_num * noise_ratio)
    noisy_idxs = idxs[:num_noise]
    noisy_labels = []
    if len(class_space) < 2:
        raise ValueError(
            f"Repartition the dataset! Each client should at least contain 2 classes!"
        )

    for i in range(sample_num):
        if i in noisy_idxs:
            if noise_mode == "sym":
                y_ = other_class(class_space, labels[i])
            elif noise_mode == "asym":
                possible_y = transition[labels[i]]
                if possible_y in class_space:
                    y_ = possible_y
                else:
                    y_ = next_class(class_space, labels[i], dataset)
            noisy_labels.append(y_)
        else:
            noisy_labels.append(labels[i])

    return noisy_labels


def split_data(client_dict: Dict[int, List[int]], data: List[Any]):
    """Split ``data`` to dictionary format according to the partition result ``client_dict``.

    Args:
        client_dict: {cid1: [idx1, ...], cid2: [idx1, ...], ...}
        data: list of data, [data1, data2, ...]. data can be any type of info.

    Returns:
    """
    data_dict = dict()
    for cid in client_dict:
        cur_idxs = client_dict[cid]
        data_dict[cid] = [data[i] for i in cur_idxs]
    return data_dict


def merge_data(
    index_dict: Dict[int, List[int]],
    data_dict: Dict[int, List[Any]],
    num_samples: Optional[int] = 50000,
) -> List[Any]:
    """Merge data_dict into complete data, which is inversion operation for function :func:``split_data``.

    Args:
        index_dict (Dict[int, List[int]]): {cid1: [idx1, ...], ...}. Contains list of data indices for each client ID.
        data_dict (Dict[int, List[Any]]): {cid1: [data1, ...], ...}. Contains list of data for each client ID. The order of data corresponds to that of ``client_id``.
        num_samples (int):

    Returns:
        List[Any]:
    """
    assert sorted(data_dict.keys()) == sorted(index_dict.keys())
    merged_result = [None for _ in range(num_samples)]
    for cid in index_dict:
        cur_index = index_dict[cid]
        for i in range(len(cur_index)):
            merged_result[cur_index[i]] = data_dict[cid][i]
    return merged_result


def generate_noisy_label_cifar100(
    labels: List[int], noise_mode: str = "sym", noise_ratio: float = 0.1, seed: int = 0
) -> List[int]:
    num_classes = CLASS_NUM["cifar100"]
    class_space = set(range(num_classes))
    noise_ratio = check_centr_noisy_setting(
        noise_mode, noise_ratio
    )  # check for validation of noisy setting
    if noise_mode == "asym":
        # mistakes are inside the same superclass of 10 classes, e.g. 'fish'
        num_superclasses = 20
        num_subclasses = 5
        P = np.eye(num_classes)
        for i in range(num_superclasses):
            init, end = i * num_subclasses, (i + 1) * num_subclasses
            P[init:end, init:end] = build_for_cifar100(num_subclasses, noise_ratio)
            noisy_labels = multiclass_noisify(np.array(labels), P=P, random_state=seed)
        noisy_labels = noisy_labels.tolist()

    else:
        #  noise_mode == "sym" or "clean"
        noisy_labels = symmetric_label_flipping(
            labels, class_space, noise_ratio=noise_ratio
        )

    return noisy_labels


def next_class(
    class_space: Set[int], current_class: int, dataset: str = "cifar10"
) -> int:
    """Return the next class index listed after the current class in the ``class_space``.

    If ``dataset`` is not ``"cifar100"``, then return the next class in ``class_space``. If ``dataset`` is ``"cifar100"``, return the next subclass in ``class_space`` if exists, otherwise return the next class in ``class_space`` like other dataset.

    Args:
        class_space (Set[int]): _description_
        current_class (int): _description_
        dataset (str, optional): _description_. Defaults to "cifar10".

    Returns:
        int: _description_
    """
    candidate_classes = sorted(list(class_space))
    if dataset == "cifar100":
        nb_subclasses = 5
        group_idx = current_class // nb_subclasses
        first_class = group_idx * nb_subclasses
        last_class = (group_idx + 1) * nb_subclasses - 1
        cur_group = set(range(first_class, last_class + 1))
        print(f"Current group: {cur_group}")
        subclass_choice = set.intersection(cur_group, class_space)
        if len(subclass_choice) > 1:
            candidate_classes = sorted(list(subclass_choice))
            print(f"subclass founded in class space: {candidate_classes}")

    cur_idx = candidate_classes.index(current_class)
    try:
        chosen_class = candidate_classes[cur_idx + 1]
    except IndexError:
        chosen_class = candidate_classes[0]

    return chosen_class


def other_class(class_space: Set[int], current_class: int):
    """Return one random class index excluding the class indexed by ``current_class``.

    Args:
        class_space (Set[int]): Set of classes, ``{1,3,5}`` for example.
        current_class (int): Class index to be omitted

    Returns:
        int, one random class index that != current_class
    """
    other_class_list = list(class_space)
    other_class_list.remove(current_class)
    chosen_class = np.random.choice(other_class_list).tolist()
    return chosen_class


def multiclass_noisify(y, P, random_state=0):
    """Flip classes according to transition probability matrix P.
    It expects a number between 0 and the number of classes - 1.

    Returns:
        np.ndarray
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def cal_true_noisy_ratio(labels: List[int], noisy_labels: List[int]) -> float:
    eq_list = map(lambda y, y_: int(y_ == y), labels, noisy_labels)
    true_noise_ratio = 1.0 - sum(eq_list) / len(labels)
    return true_noise_ratio


def cal_multiple_true_noisy_ratio(
    labels_dict: Dict[int, List[int]], noisy_labels_dict: Dict[int, List[int]]
) -> Dict[int, float]:
    true_noisy_ratio_dict = dict()
    assert sorted(labels_dict.keys()) == sorted(noisy_labels_dict.keys())
    for cid in noisy_labels_dict:
        true_noisy_ratio_dict[cid] = cal_true_noisy_ratio(
            labels_dict[cid], noisy_labels_dict[cid]
        )
    return true_noisy_ratio_dict


def unpickle(file):
    import _pickle as cPickle

    with open(file, "rb") as fo:
        entry = cPickle.load(fo, encoding="latin1")
    return entry


# TRANSITION_MATRIX = {
#     "cifar10": {0: 0, 1: 1, 2: 0, 3: 5, 4: 7, 5: 3, 6: 6, 7: 7, 8: 8, 9: 1},
#     "cifar100": build_cifar100_transmat(),
#     "mnist": {0: 0, 1: 1, 2: 7, 3: 8, 4: 4, 5: 6, 6: 5, 7: 1, 8: 8, 9: 9},
#     "svhn": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0},
# }

TRANSITION_MATRIX = {
    "cifar10": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0},
    "cifar100": build_cifar100_transmat(),
    "mnist": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0},
    "svhn": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0},
}
