# Distilling Dataset into Generative Models

Official implementation of "[DiM: Distilling Dataset into Generative Model](https://arxiv.org/abs/2303.04707)". 

## Abstract

Dataset distillation reduces the network training cost by synthesizing small and informative datasets from largescale ones. Despite the success of the recent dataset distillation algorithms, three drawbacks still limit their wider application: i). The synthetic images perform poorly on large architectures; ii). They need to be re-optimized when the distillation ratio changes; iii). The limited diversity restricts the performance when the distillation ratio is large. In this paper, we propose a novel distillation scheme to Distill information of large train sets into generative Models, named DiM. Specifically, DiM learns to use a generative model to store the information of the target dataset. During the distillation phase, we minimize the differences in logits predicted by a models pool between real and generated images. At the deployment stage, the generative model synthesizes various training samples from random noises on the fly. Due to the simple yet effective designs, the trained DiM can be directly applied to different distillation ratios and large architectures without extra cost. We validate the proposed DiM across 4 datasets and achieve state-of-the-art results on all of them. To the best of our knowledge, we are the first to achieve higher accuracy on complex architectures than simple ones, such as 75.1% with ResNet-18 and 72.6% with ConvNet-3 on ten images per class of CIFAR-10. Besides, DiM outperforms previous methods with 10% ∼ 22% when images per class are 1 and 10 on the SVHN dataset.

![pipeline](figs\pipeline.png)

## Datasets

* CIFAR-10
* SVHN
* MNIST
* Fashion-MNIST

The datasets will be downloaded at the first running time. 

## Experiment Commands

Train a conditional GAN model.

```bash
CUDA_VISIBLE_DEVICES=0 python gen_condense.py --tag gan
```

Select the best model for further matching.

```bash
CUDA_VISIBLE_DEVICES=0 python pool_match.py --tag match --match logit --match-aug --weight [BEST_MODEL]
```

Validate the generator performance.

```bash
CUDA_VISIBLE_DEVICES=0 python validate.py --tag test --pretrain-weight [BEST_MODEL]
```

## Citation

```
@article{wang2023dim,
  title={DiM: Distilling Dataset into Generative Model},
  author={Wang, Kai and Gu, Jianyang and Zhou, Daquan and Zhu, Zheng and Jiang, Wei and You, Yang},
  journal={arXiv preprint arXiv:2303.04707},
  year={2023}
}
```
