# FedNoisy: Federated Noisy Label Learning Benchmark

Benchmark for federated noisy label learning.


[toc]

[![License](https://img.shields.io/github/license/SMILELab-FL/FedNoisy)](https://opensource.org/licenses/Apache-2.0) [![arXiv](https://img.shields.io/badge/arXiv-2306.11650-red.svg)](https://arxiv.org/abs/2306.11650) 

Federated learning has gained popularity for distributed learning without aggregating sensitive data from clients. But meanwhile, the distributed and isolated nature of data isolation may be complicated by data quality, making it more vulnerable to noisy labels. Many efforts exist to defend against the negative impacts of noisy labels in centralized or federated settings. However, there is a lack of a benchmark that comprehensively considers the impact of noisy labels in a wide variety of typical FL settings. 

In this work, we serve the first standardized benchmark that can help researchers fully explore potential federated noisy settings. Also, we conduct comprehensive experiments to explore the characteristics of these data settings and unravel challenging scenarios on the federated noisy label learning, which may guide method development in the future. We highlight the 20 basic settings for more than 5 datasets proposed in our benchmark and standardized simulation pipeline for federated noisy label learning. We hope this benchmark can facilitate idea verification in federated learning with noisy labels. 

## Environment requirements

Pytorch installation via `conda`:

```bash
$ conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c nvidia -y
```

Extra dependencies:
```bash
$ pip install -r requirements.txt
```



## Federated Noisy Labe Settings

Current supported datasets:
<table>
	<tbody>
		<tr >
			<th style="text-align: left; vertical-align: middle;">Noise Scene</th>
			<th style="text-align: left; vertical-align: middle;">Dataset</th>
			<th style="text-align: left; vertical-align: middle;">#Train</th>
			<th style="text-align: left; vertical-align: middle;">#Validation</th>
			<th style="text-align: left; vertical-align: middle;">#Test</th>
			<th style="text-align: left; vertical-align: middle;">#Class</th>
			<th style="text-align: left; vertical-align: middle;">ImageSize</th>
			<th style="text-align: left; vertical-align: middle;">Noise Ratio (%)</th>
		</tr>
		<tr >
			<td style="vertical-align: middle;" rowspan="4">
				<p>Globalized</p>
				<p>&amp;</p>
				<p>Localized</p>
			</td>
			<td style="text-align: left; vertical-align: middle;">MNIST</td>
			<td>60K</td>
			<td>N/A</td>
			<td>10K</td>
			<td>10</td>
			<td>28&times;28</td>
			<td>N/A</td>
		</tr>
		<tr >
			<td style="text-align: left; vertical-align: middle;">SVHN</td>
			<td>73K</td>
			<td>N/A</td>
			<td>26K</td>
			<td>10</td>
			<td>32&times;32&times;3</td>
			<td>N/A</td>
		</tr>
		<tr >
			<td style="text-align: left; vertical-align: middle;">CIFAR-10</td>
			<td>50K</td>
			<td>N/A</td>
			<td>10K</td>
			<td>10</td>
			<td>32&times;32&times;3</td>
			<td>N/A</td>
		</tr>
		<tr >
			<td style="text-align: center; vertical-align: middle;">CIFAR-100</td>
			<td>50K</td>
			<td>N/A</td>
			<td>10K</td>
			<td>100</td>
			<td>32&times;32&times;3</td>
			<td>N/A</td>
		</tr>
		<tr >
			<td style="text-align: center; vertical-align: middle;" rowspan="2">Real-world</td>
			<td style="text-align: center; vertical-align: middle;">Clothing1M</td>
			<td>1M</td>
			<td>14K</td>
			<td>10K</td>
			<td>14</td>
			<td>224&times;224&times;3</td>
			<td>&asymp;39.46</td>
		</tr>
		<tr >
			<td>WebVision</td>
			<td>2.4M</td>
			<td>50K</td>
			<td>50K</td>
			<td>1000</td>
			<td>256&times;256&times;3</td>
			<td>&asymp;20.00</td>
		</tr>
	</tbody>
</table>


Federated noise scenes provided in $\texttt{FedNoisy}$ is shown as below. (Left: _globalized noise_; middle: _localized noise_; right: _real-word noise_. Globalized noise follows a constant noise ratio $\varepsilon_{global}$ in label corruption, while localized noise draws localized noise ratio $\varepsilon_{k}$ from a uniform distribution on each client $k$.)

<img src="./imgs/fednoisy-noise-scenes.png" height="400">



### Prepare

Raw dataset should be downloaded in to local folder before data-build process. The folder structure can be:

```shell
.
├── FedNoisy
│   ├── LICENSE
│   ├── README.md
│   ├── build_dataset_fed.py
│   ├── fednoisy
│   ├── imgs
│   ├── requirements.txt
│   └── scripts
├── rawdata
│   ├── cifar10
│   │   ├── cifar-10-batches-py
│   │   └── cifar-10-python.tar.gz
│   ├── cifar100
│   │   ├── cifar-100-python
│   │   └── cifar-100-python.tar.gz
│   ├── mnist
│   │   └── MNIST
│   │       └── raw
│   │           ├── t10k-images-idx3-ubyte
│   │           ...
│   │           └── train-labels-idx1-ubyte.gz
│   ├── clothing1M/
│   │   ├── category_names_chn.txt
│   │   ├── category_names_eng.txt
│   │   ├── clean_label_kv.txt
│   │   ├── clean_test_key_list.txt
│   │   ├── clean_train_key_list.txt
│   │   ├── clean_val_key_list.txt
│   │   ├── images
│   │   │   ├── 0
│   │   │   ├── 1
│   │   │   ...
│   │   │   └── 9
│   │   ├── noisy_label_kv.txt
│   │   ├── noisy_train_key_list.txt
│   │   ├── README.md
│   │   └── venn.png
│   └── svhn
│       ├── extra_32x32.mat
│       ├── test_32x32.mat
│       └── train_32x32.mat
├── fedNLLdata  # to store fedNLL data settings 
└── Fed-Noisy-checkpoint  # to store algorithm logging output and final models 
```

- Use `torchvision.dataset` to download of __CIFAR-10__/__CIFAR-100__/__SVHN__/__MNIST__ directly:

  ```python
  import torchvision
  
  mnist_train = torchvision.datasets.MNIST(root="rawdata/mnist", train=True, download=True)
  mnist_test = torchvision.datasets.MNIST(root="rawdata/mnist", train=False, download=True)
  
  cifar10_train = torchvision.datasets.CIFAR10(root="rawdata/cifar10", train=True, download=True)
  cifar10_test = torchvision.datasets.CIFAR10(root="rawdata/cifar10", train=False, download=True)
  
  cifar100_train = torchvision.datasets.CIFAR100(root="rawdata/cifar100", train=True, download=True)
  cifar100_test = torchvision.datasets.CIFAR100(root="rawdata/cifar100", train=False, download=True)
  
  svhn_train = torchvision.datasets.SVHN(root="rawdata/svhn", split="train", download=True)
  svhn_test = torchvision.datasets.SVHN(root="rawdata/svhn", split="test", download=True)
  ```

- To download __[Clothing1M](https://github.com/Cysu/noisy_label)__, please contact *tong.xiao.work[at]gmail[dot]com* to get the download link. Untar the images and unzip the annotations under `rawdata/clothing1M`.



### Build dataset with noisy label

The basic command usage is

```bash
$ cd FedNoisy 

# under dir FedNoisy/
$ python build_dataset_fed.py --dataset cifar10 \
     --partition iid \
     --num_clients 10 \
     --globalize \
     --noise_mode clean \
     --raw_data_dir ../rawdata/cifar10 \
     --data_dir ../fedNLLdata/cifar10 \
     --seed 1
```



#### Noise related setting

- Clean: `--globalize --noise_mode clean` for data setting without noise

- Globalized noise
  - `--globalize --noise_ratio 0.4 --noise_mode sym` for globalized symmetric noise $\varepsilon_{global}=0.4$
  - `--globalize --noise_ratio 0.4 --noise_mode asym` for globalized asymmetric noise $\varepsilon_{global}=0.4$
  
- Localized noise 
  - `--min_noise_ratio 0.3 --max_noise_ratio 0.5 --noise_mode sym` for localized symmetric noise $\varepsilon_k \sim \mathcal{U}(0.3, 0.5)$
  - `--min_noise_ratio 0.3 --max_noise_ratio 0.5 --noise_mode asym` for localized asymmetric noise $\varepsilon_k \sim \mathcal{U}(0.3, 0.5)$
  
- Real noise (only works for Clothing1M): `--dataset clothing1m --globalize --noise_mode real --num_sampels 64000`

  - `--num_samples` is for specifying number of training samples used for Clothing1M, the default is 64000

  



#### Dataset partition setting

- MNIST: `--dataset mnist`

  - IID: `--partition iid --num_clients 10`
  - Non-IID quantity skew: `--partition noniid-quantity --num_clients 10 --dir_alpha 0.1`
  - Non-IID Dirichlet-based label skew: `--partition noniid-labeldir --dir_alpha 0.1 --num_clients 10`
  - Non-IID quantity-based label skew: `--partition noniid-#label --major_classes_num 3 --num_clients 10`

- SVHN: `--dataset svhn`

- CIFAR-10: `--dataset cifar10`

- CIFAR-100:  `--dataset cifar100`

- Clothing1M: `--dataset cloting1m`
  - IID: `--partition iid --num_clients 10`
  - Non-IID quantity skew: `--partition noniid-quantity --num_clients 10 --dir_alpha 0.1`
  - Non-IID Dirichlet-based label skew: `--partition noniid-labeldir --dir_alpha 0.1 --num_clients 10`
  - Non-IID quantity-based label skew: `--partition noniid-#label --major_classes_num 5 --num_clients 10`






### Algorithms

<table style="margin-left: auto; margin-right: auto;">
<tbody>
<tr>
<td  rowspan="2"><strong>Federated Algorithm</strong></td>
<td  colspan="2"><strong>Noisy Label Algorithm</strong></td>
<td  rowspan="2"><strong>Paper</strong></td>
</tr>
<tr>
<td><em>Category</em></td>
<td><em>Method</em></td>
</tr>
<tr>
<td  rowspan="8">FedAvg</td>
<td>Robust Regularizaiton</td>
<td>Mixup</td>
<td><a href="https://arxiv.org/abs/1710.09412" target="_blank">[2018 ICLR] Mixup: Beyond empirical risk minimization</a></td>
</tr>
<tr>
<td  rowspan="3">RobustLoss Function</td>
<td>SCE</td>
<td><a href="https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.pdf" target="_blank">[2019 ICCV] Symmetric Cross Entropy for Robust Learning with Noisy Labels</a></td>
</tr>
<tr>
<td>GCE</td>
<td><a href="https://proceedings.neurips.cc/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html" target="_blank">[2018 NeurIPS] Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels</a></td>
</tr>
<tr>
<td>MAE</td>
<td><a href="https://arxiv.org/abs/1712.09482" target="_blank">[2017 AAAI] Robust Loss Functions under Label Noise for Deep Neural Networks</a></td>
</tr>
<tr>
<td  rowspan="3">Loss Adjustment</td>
<td>M-DYR-H</td>
<td><a href="http://proceedings.mlr.press/v97/arazo19a/arazo19a.pdf" target="_blank">[2019 ICML] Unsupervised Label Noise Modeling and Loss Correction</a></td>
</tr>
<tr>
<td>M-DYR-S</td>
<td><a href="http://proceedings.mlr.press/v97/arazo19a/arazo19a.pdf" target="_blank">[2019 ICML] Unsupervised Label Noise Modeling and Loss Correction</a></td>
</tr>
<tr>
<td>DM-DYR-SH</td>
<td><a href="http://proceedings.mlr.press/v97/arazo19a/arazo19a.pdf" target="_blank">[2019 ICML] Unsupervised Label Noise Modeling and Loss Correction</a></td>
</tr>
<tr>
<td>Sample Selection</td>
<td>Co-teaching</td>
<td><a href="https://arxiv.org/abs/1804.06872" target="_blank">[2018 NeurIPS] Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels</a></td>
</tr>
</tbody>
</table>





- __FedAvg__

  - Vanilla FedAvg

    ```bash
    # under dir FedNoisy/
    $ python python fednoisy/algorithms/fedavg/main.py --dataset mnist \
        --model SimpleCNN \
        --partition iid \
        --num_clients 10 \
        --globalize \
        --noise_mode sym \
        --noise_ratio 0.4 \
        --seed 1 \
        --sample_ratio 1.0 \
        --com_round 500 \
        --epochs 5 \
        --momentum 0.9 \
        --lr 0.01 \
        --weight_decay 0.0005 \
        --data_dir ../fedNLLdata/mnist \
        --out_dir ../Fed-Noisy-checkpoint/mnist/
    ```
  
  - FedAvg + Symmetric Cross Entropy
  
    ```bash
    # under dir FedNoisy/
    $ python fednoisy/algorithms/fedavg/main.py --dataset mnist \
        --model SimpleCNN \
        --partition iid \
        --num_clients 10 \
        --globalize \
        --noise_mode sym \
        --noise_ratio 0.4 \
        --data_dir ../fedNLLdata/mnist \
        --out_dir ../Fed-Noisy-checkpoint/mnist/ \
        --com_round 500 \
        --epochs 5 \
        --sample_ratio 1.0 \
        --lr 0.01 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --criterion sce \
        --sce_alpha 0.01 \
        --sce_beta 1.0 \
        --seed 1
    ```
  
  - FedAvg + GCE
  
    ```bash
    # under dir FedNoisy/
    $ python fednoisy/algorithms/fedavg/main.py --dataset mnist \
        --model SimpleCNN \
        --partition iid \
        --num_clients 10 \
        --globalize \
        --noise_mode sym \
        --noise_ratio 0.4 \
        --data_dir ../fedNLLdata/mnist \
        --out_dir ../Fed-Noisy-checkpoint/mnist/ \
        --com_round 500 \
        --epochs 5 \
        --sample_ratio 1.0 \
        --lr 0.01 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --criterion gce \
        --gce_q 0.7 \
        --seed 1
    ```
  
  - FedAvg + MAE
  
    ```bash
    # under dir FedNoisy/
    $ python fednoisy/algorithms/fedavg/main.py --dataset mnist \
        --model SimpleCNN \
        --partition iid \
        --num_clients 10 \
        --globalize \
        --noise_mode sym \
        --noise_ratio 0.4 \
        --data_dir ../fedNLLdata/mnist \
        --out_dir ../Fed-Noisy-checkpoint/mnist/ \
        --com_round 500 \
        --epochs 5 \
        --sample_ratio 1.0 \
        --lr 0.01 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --criterion mae \
        --loss_scale 1.0 \
        --seed 1
    ```
  
  - FedAvg + M-DYR-H
  
    ```bash
    # under dir FedNoisy/
    $ python fednoisy/algorithms/fedavg/main.py --dataset mnist \
        --model SimpleCNN \
        --partition iid \
        --num_clients 10 \
        --globalize \
        --noise_mode sym \
        --noise_ratio 0.4 \
        --data_dir ../fedNLLdata/mnist \
        --out_dir ../Fed-Noisy-checkpoint/mnist/ \
        --com_round 500 \
        --epochs 5 \
        --sample_ratio 1.0 \
        --lr 0.01 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --dynboot \
        --dynboot_alpha 32 \
        --dynboot_mixup static \
        --dynboot_bootbeta hard \
        --dynboot_reg 1.0 \
        --seed 1
    ```
  
  - FedAvg + M-DYR-S
  
    ```bash
    # under dir FedNoisy/
    $ python fednoisy/algorithms/fedavg/main.py --dataset mnist \
        --model SimpleCNN \
        --partition iid \
        --num_clients 10 \
        --globalize \
        --noise_mode sym \
        --noise_ratio 0.4 \
        --data_dir ../fedNLLdata/mnist \
        --out_dir ../Fed-Noisy-checkpoint/mnist/ \
        --com_round 500 \
        --epochs 5 \
        --sample_ratio 1.0 \
        --lr 0.01 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --dynboot \
        --dynboot_alpha 32 \
        --dynboot_mixup static \
        --dynboot_bootbeta soft \
        --dynboot_reg 1.0 \
        --seed 1
    ```
  
  - FedAvg + DM-DYR-SH
  
    ```bash
    # under dir FedNoisy/
    $ python fednoisy/algorithms/fedavg/main.py --dataset mnist \
        --model SimpleCNN \
        --partition iid \
        --num_clients 10 \
        --globalize \
        --noise_mode sym \
        --noise_ratio 0.4 \
        --data_dir ../fedNLLdata/mnist \
        --out_dir ../Fed-Noisy-checkpoint/mnist/ \
        --com_round 500 \
        --epochs 5 \
        --sample_ratio 1.0 \
        --lr 0.01 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --dynboot \
        --dynboot_alpha 32 \
        --dynboot_mixup dynamic \
        --dynboot_reg 1.0 \
        --seed 1
    ```
  
  - FedAvg + Mixup
  
    ```bash
    # under dir FedNoisy/
    $ python fednoisy/algorithms/fedavg/main.py --dataset mnist \
        --model SimpleCNN \
        --partition iid \
        --num_clients 10 \
        --globalize \
        --noise_mode sym \
        --noise_ratio 0.4 \
        --data_dir ../fedNLLdata/mnist \
        --out_dir ../Fed-Noisy-checkpoint/mnist/ \
        --com_round 500 \
        --epochs 5 \
        --sample_ratio 1.0 \
        --lr 0.01 \
        --momentum 0.9 \
        --weight_decay 5e-4 \
        --mixup \
        --mixup_alpha 1.0 \
        --seed 1
    ```
  
  - FedAvg + Co-teaching
  
    ```bash
    # under dir FedNoisy/
    $ python fednoisy/algorithms/fedavg/main.py --dataset mnist \
        --model SimpleCNN \
        --partition iid \
        --num_clients 10 \
        --globalize \
        --noise_mode sym \
        --noise_ratio 0.4 \
        --data_dir ../fedNLLdata/mnist  \
        --out_dir ../Fed-Noisy-checkpoint/mnist/ \
        --com_round 500 \
        --epochs 5 \
        --sample_ratio 1.0 \
        --lr 0.01 \
        --momentum 0.9 \
        --weight_decay 5e-4 \
        --coteaching \
        --coteaching_forget_rate 0.4 \  # depends on noise ratio, possible value can be noise ratio it self 
        --coteaching_num_gradual 25 \  # default setting is 10 for 200 epochs, here we set similar ratio with number of global round
        --seed 1
    ```
    
    

For more scripts, please check [scripts](./scripts/) folder.



## Development plan

### Datasets

- [ ] Include mini-ImageNet (synthetic noise)
- [ ] Include Food-101N (real-world noise)
- [ ] Include ANIMAL-10N (real-world noise)



### FNLL baselines

- [ ] Implement FedNoRo (working on...)
- [ ] Implement FedNed (working on...)
- [ ] Implement FedCorr (a little bit complicated)



### Others

- [ ] Mixed noise modes over different clients (easy to implement)
- [ ] Text datasets (long term goal)
- [ ] Graph node classification datasets (long term goal)



## Contributing to FedNoisy

### Reporting bugs

We use GitHub issues to track all bugs and feature requests. Feel free to open an issue if you have found a bug or wish to see a feature implemented.

In case you experience issues using this package, do not hesitate to submit a ticket to the [Bug Tracker](https://github.com/SMILELab-FL/FedNoisy/issues). You are also welcome to post feature requests or pull requests.



### Contributing code

You’re welcome to contribute to this project through **Pull Request**. By contributing, you agree that your contributions will be licensed under [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.html).



We encourage you to contribute to the improvement of FedNoisy or the FedNoisy implementation of existing FNLL methods. The preferred workflow for contributing to FedNoisy is to fork the main repository on GitHub, clone, and develop on a branch. Steps as follow:

1. Fork the project repository by clicking on the `Fork`. For contributing new features, please fork FedNoisy [core repo](https://github.com/SMILELab-FL/FedNoisy).

2. Clone your fork of repo from your GitHub to your local:

   ```shell
   $ git clone git@github.com:YourLogin/FedLab.git
   $ cd FedNoisy
   ```

3. Create a new branch to save your changes:

   ```shell
   $ git checkout -b my-feature
   ```

4. Develop the feature on your branch.

   ```shell
   $ git add modified_files
   $ git commit
   ```



### Pull request checklist

- Please follow the file structure below for new features or create new file if there are something new.

  ```shell
  fednoisy
  ├── __init__.py
  ├── algorithms
  │   ├── __init__.py
  │   └── fedavg
  │       ├── __init__.py
  │       ├── client.py               # FL client class definition
  │       ├── main.py                 # FL algorithm main script
  │       ├── misc.py                 
  │       ├── server.py               # FL server class definition
  │       └── standalone.py           # FL standalone pipeline class definition
  ├── data
  │   ├── NLLData
  │   │   ├── BaseNLL                 # Base class for datasets, for download and preprocess
  │   │   ├── CentrNLL                # CNLL class for datasets, including adding noise 
  │   │   ├── FedNLL                  # FNLL class for datasets, including adding noise and partition
  │   │   ├── __init__.py
  │   │   └── functional.py           # Utility functions for noisy label datasets
  │   ├── __init__.py
  │   ├── dataset.py
  │   ├── ...
  │   └── functional.py
  ├── models
  │   ├── __init__.py
  │   ├── build_model.py               # model entries
  │   ├── cnn.py                       # model architecture files
  │   ├── .....
  │   └── wideresnet.py
  ├── utils                            # Utitlity functions for different algorithms  
  └── visual.py                        # Visualization functions for FNLL & CNLL data
  ```

- Please provide shell script to launch new functions or new algorithms

- All public methods should have informative docstrings with sample usage presented as doctests when appropriate. Docstring and code should follow Google Python Style Guide: [中文版](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/) | [English](https://google.github.io/styleguide/pyguide.html).

  > __Note:__ We understand it can be a burden to add detailed docstring for code. Please have a try if it is convenient for you.



## Citation

Please cite __FedNoisy__ in your publications if it helps your research:

```latex
@article{liang2023fednoisy,
  title={FedNoisy: Federated noisy label learning benchmark},
  author={Liang, Siqi and Huang, Jintao and Hong, Junyuan and Zeng, Dun and Zhou, Jiayu and Xu, Zenglin},
  journal={arXiv preprint arXiv:2306.11650},
  year={2023}
}
```




## References

<a id="1">[1]</a> LeCun, Y. (1998). The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/.

<a id="2">[2]</a> Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.

<a id="3">[3]</a> Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., & Ng, A. Y. (2011). Reading digits in natural images with unsupervised feature learning.

<a id="4">[4]</a> Xiao, T., Xia, T., Yang, Y., Huang, C., & Wang, X. (2015). Learning from massive noisy labeled data for image classification. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2691-2699).

<a id="5">[5]</a> Li, W., Wang, L., Li, W., Agustsson, E., & Van Gool, L. (2017). Webvision database: Visual learning and understanding from web data. arXiv preprint arXiv:1708.02862.

