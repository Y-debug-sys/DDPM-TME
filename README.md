# Official Implementation: "DDPM-Based Methods for Future Traffic Matrix Estimation and Synthesis"

This is the github repoistory for an accurate DDPM-based traffic estimation approach implemented in pytorch. The implementation was transcribed from the Pytorch version of DDPM <a href="https://github.com/lucidrains/denoising-diffusion-pytorch">here</a>

Paper: 

## Abstract

> Traffic matrix estimation (TME) problem has been widely researched for decades of years. Recent progresses in deep generative models offer new opportunities to tackle TME problem in a more advanced way. In this paper, we leverage the powerful ability of denoising diffusion probabilistic models (DDPMs) on distribution learning, and for the first time adopt DDPM to address the TME problem. To ensure a good performance of DDPM on learning the distributions of TMs, we design a preprocessing module to reduce the dimensions of TMs while keep the data variety of each OD flow. To improve the estimation accuracy, we parameterize the noise factors in DDPM and transform the TME problem into a gradient-descent optimization problem. Finally, we compared our method with the state-of-the-art TME methods using two real-world TM datasets, the experimental results strongly demonstrate the superiority of our method on both TM synthesis and TM estimation.

## Requirements

This project has been developed and tested in Python 3.7.13 and requires the following libraries:

- Numpy==1.21.6
- Pandas==1.3.5
- Matplotlib==3.6.0

## Framework

- Pytorch==1.12.0

## Datasets

- Abilene Dataset
- G´EANT Dataset

## Methods

- DDPM-TME
- VAE https://github.com/MikeKalnt/VAE-TME

## File structure:



## Example args:



## How to use

Simple usage:
```bash
python main.py
```

## Author

- Yan Qiao -  [qiaoyan@hfut.edu.cn](qiaoyan:qiaoyan@hfut.edu.cn)
- Xinyu Yuan - [2022111103@mail.hfut.edu.cn](yuanxinyu:2022111103@mail.hfut.edu.cn) / [yxy5315@gmail.com](yuanxinyu:yxy5315@gmail.com)

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

Distributed under the MIT License. See [LICENCE](https://github.com/Y-debug-sys/AutoTomo/blob/main/LICENSE) for more information.
