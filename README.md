# Official Implementation: "DDPM-Based Methods for Future Traffic Matrix Estimation and Synthesis"

This is the github repoistory for an accurate DDPM-based traffic estimation approach implemented in pytorch. The implementation was transcribed from the Pytorch version of <a href="https://github.com/lucidrains/denoising-diffusion-pytorch">DDPM</a>

Paper: 

<p align="center">
  <img src='https://github.com/Y-debug-sys/DDPM-TME//blob/overview.jpg' width=45%>
</p>

## Abstract

> Traffic matrix estimation (TME) problem has been widely researched for decades of years. Recent progresses in deep generative models offer new opportunities to tackle TME problem in a more advanced way. In this paper, we leverage the powerful ability of denoising diffusion probabilistic models (DDPMs) on distribution learning, and for the first time adopt DDPM to address the TME problem. To ensure a good performance of DDPM on learning the distributions of TMs, we design a preprocessing module to reduce the dimensions of TMs while keep the data variety of each OD flow. To improve the estimation accuracy, we parameterize the noise factors in DDPM and transform the TME problem into a gradient-descent optimization problem. Finally, we compared our method with the state-of-the-art TME methods using two real-world TM datasets, the experimental results strongly demonstrate the superiority of our method on both TM synthesis and TM estimation.

## Requirements

This project has been developed and tested in Python 3.7.13 and requires the following libraries:

- Numpy==1.21.6
- Pandas==1.3.5
- Matplotlib==3.6.0
- Tqdm==4.64.1
- Einops==0.6.0

## Framework

- Pytorch==1.12.0

## Datasets

- Abilene dataset
- G´EANT dataset

## Methods

- DDPM-TME
- VAE-TME - https://github.com/MikeKalnt/VAE-TME
- WGAN-GP - https://github.com/caogang/wgan-gp

## File structure:

- Dataset/ - CSV files of remote martrix and traffic matrix sampled from Abilene and G´EANT dataset
- cfg.py - The procedure defines what arguments it requires
- model.py - UNet, Embedding-Recovery and Gaussian DDPM architecture, forked from https://github.com/lucidrains/denoising-diffusion-pytorch
- train.py - Training procedure
- estimate.py - Traffic matrix estimation based on gradient-descent optimization
- utils.py - Helper functions for custom dataset loader, earlystopping and plotting
- main.py - Main function

## Example args:

{
"batch_size": 64,
"concur_size": 100,
"loss_func_1": "l1",
"loss_func_2": "l1",
"dataset": "abilene",
"hd": 32,
"dim_mults": (1, 2, 4),
"tt": 1000,
"st": 200,
"schedule": "cosine",
"visualize": "tsne",
}

## How to use

Simple usage:
```bash
python-m main.py
```

## Author

- Xinyu Yuan - [2022111103@mail.hfut.edu.cn](yuanxinyu:2022111103@mail.hfut.edu.cn) / [yxy5315@gmail.com](yuanxinyu:yxy5315@gmail.com)
- Yan Qiao -  [qiaoyan@hfut.edu.cn](qiaoyan:qiaoyan@hfut.edu.cn)

<!-- ## Citation

```
@InProceedings{Yanqiao23,
    author    = {Yan Qiao, Qui Wu, Xinyu Yuan.},
    title     = {AutoTomo: Accurate and Low-cost Traffic Estimator Integrating Network Tomography},
    booktitle = {International Conference on Distributed Computing Systems (ICDCS)},
    month     = {June},
    year      = {2023},
    pages     = {-}
}
``` -->

## Licence

Distributed under the MIT License. See [LICENCE](https://github.com/Y-debug-sys/DDPM-TME/blob/main/LICENSE) for more information.
