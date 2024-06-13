# MimicDiffusion: Purifying Adversarial Perturbation via Mimicking Clean Diffusion Model
Official implement for MimicDiffusion with Pytorch      
> Kaiyu Song, Hanjiang Lai, Yan Pan, Jian Yin    
> *arXiv technical report ([arXiv 2312.04802](https://arxiv.org/abs/2312.04802))*


## Todo
- [ ] release the rest code for CIFAR-10 based on the adjoint method
- [ ] release the code for ImageNet based on the surrogate process
- [ ] release the code for CIFAR-10 based on the surrogate process using EDM
- [x] release the code for CIFAR-10 based on the surrogate process using SDE 
## Abstract

Deep neural networks (DNNs) are vulnerable to adversarial perturbation, where an imperceptible perturbation is added to the image that can fool the DNNs. Diffusion-based adversarial purification uses the diffusion model to generate a clean image against such adversarial attacks. Unfortunately, the generative process of the diffusion model is also inevitably affected by adversarial perturbation since the diffusion model is also a deep neural network where its input has adversarial perturbation. In this work, we propose MimicDiffusion, a new diffusion-based adversarial purification technique that directly approximates the generative process of the diffusion model with the clean image as input. Concretely, we analyze the differences between the guided terms using the clean image and the adversarial sample. After that, we first implement MimicDiffusion based on Manhattan distance. Then, we propose two guidance to purify the adversarial perturbation and approximate the clean diffusion model. 
Extensive experiments on three image datasets, including CIFAR-10, CIFAR-100, and ImageNet, with three classifier backbones including WideResNet-70-16, WideResNet-28-10, and ResNet-50 demonstrate that MimicDiffusion significantly performs better than the state-of-the-art baselines. On CIFAR-10, CIFAR-100, and ImageNet, it achieves 92.67\%, 61.35\%, and 61.53\% average robust accuracy, which are 18.49\%, 13.23\%, and 17.64\% higher, respectively. The code is available at https://github.com/psky1111/MimicDiffusion.

## Introduction
This repo currently is based on [Surrogate Process](https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification). We mainly provide the following functionality:
+ The purification method of MimicDiffusion.


Here are some important concepts related to our project:

- **The type of gradient method for adversarial attack**: The gradient for the diffusion model is hard to calculate. [Diffpure](https://github.com/NVlabs/DiffPure) proposed the adjoint method to estimate the gradient. [Surrogate process](https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification) proposed by Lee is an improved method to calculate the gradient of the diffusion model. We highly recommend using surrogate process to implement the white-box adversarial attack since it can get more precise gradient.
- **Objective**: Our primary goal is to generate the clean image to improve the accuracy of the classifier given an adversarial sample as the condition. 

Regarding the code structure:

Our method mainly is in purification.py



## Pre-trained Models


The pre-trained diffusion model: 
- [256x256_diffusion_uncond.pt] for ImageNet (https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) from [guided-diffusion](https://github.com/openai/guided-diffusion).
- [Score SDE](https://github.com/yang-song/score_sde_pytorch) for CIFAR-10: (`vp/cifar10_ddpmpp_deep_continuous`: [download link](https://drive.google.com/file/d/16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif/view?usp=sharing)).
- [EDM](https://github.com/NVlabs/edm) for CIFAR-10.
- [EDM](https://github.com/NVlabs/edm) for CIFAR-100. We trained an EDM for CIFAR-100 using the script of the official EDM repo using the EDM checkpoint trained in CIFAR-10.



The pre-trained classifier:
- WideResNet-28-10: The pre-trained model is directly from [RobustBench](https://robustbench.github.io/)
- WideResNet-70-16: We trained a WideResNet-70-16 based on the [repo](https://github.com/meliketoy/wide-resnet.pytorch)
- ResNet-50: The pre-trained model is directly from [Pytorch](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)


## Usage
```bash
bash ./script/cifar10.sh
```



## Results

| Classifier    | Attack Method   | Gradient Method | Diffusion Model | Standard Acc | Robust Acc|
|:---------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| WideResNet-28-10    | [PGD](https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification)  | Adjoint method  | EDM | 92.05 $\pm$  6.02   | 91.55 $\pm$ 6.84  |
| WideResNet-28-10    | [PGD](https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification)  | Surrogate Process |SDE| 91.41 $\pm$ 1.12   | 80.86 $\pm$ 1.48  |





## Acknowledgments
Our work is standing on the shoulders of giants. We want to thank the following contributors that our code is based on:

Surrogate process based adversarial attack method: https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification

Adjoint method based adversarial attack method: https://github.com/NVlabs/DiffPure#requirements




## Citation

If our code or models help your work, please cite our [paper](https://arxiv.org/abs/2312.04802):
```BibTeX
@article{song2023mimicdiffusion,
  title={MimicDiffusion: Purifying Adversarial Perturbation via Mimicking Clean Diffusion Model},
  author={Kaiyu Song, Hanjiang Lai, Yan Pan, and Jian Yin},
    year={2023},
    eprint={2312.04802},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
