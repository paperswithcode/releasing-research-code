
# Selective Focusing Learning for Conditional GANs

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 


![distribution_overview1](https://user-images.githubusercontent.com/36159663/120271077-9d763080-c2e5-11eb-90cd-167ae185f0bc.png)


### TODO
⬜️ Release pretrained model on ImageNet 64x64, 128x128, CIFAR-10, CIFAR-100 dataset

⬜️ Release training code for Exact Selective Focusing Learning

## About Selective Focusing Learing

Conditional generative adversarial networks (cGANs) have demonstrated remarkable success due to their class-wise controllability and superior quality for complex generation tasks. Typical cGANs solve the joint distribution matching problem by decomposing two easier sub-problems: marginal matching and conditional matching. From our toy experiments, we found that it is the best to apply only conditional matching to certain samples due to the content-aware optimization of the discriminator. This paper proposes a simple (a few lines of code) but effective training methodology, selective focusing learning, which enforces the discriminator and generator to learn easy samples of each class rapidly while maintaining diversity. Our key idea is to selectively apply conditional and joint matching for the data in each mini-batch. We conducted experiments on recent cGAN variants in ImageNet (64x64 and 128x128), CIFAR-10, and CIFAR-100 datasets, and improved the performance significantly (up to 35.18% in terms of FID) without sacrificing diversity.
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training BigGAN with Selective Focusing Learing on ImageNet

To train BigGAN models we use the [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) and [Instance Selection for GANs](https://github.com/uoguelph-mlrg/instance_selection_for_gans) repo. We perform minimal changes to the code. The change logs are as follows. The main change part is the conditional term of the projection discriminator in BigGAN.py (L391-L402, L415-L447). Further, updating the focusing rate is represented in train.py (L66-L71, L146-L155, L185-L209).


#### Preparing Data (Same as [Instance Selection for GANs](https://github.com/uoguelph-mlrg/instance_selection_for_gans))
To train a BigGAN on ImageNet you will first need to construct an HDF5 dataset file for ImageNet (optional), compute Inception moments for calculating FID, and construct the image manifold for calculating Precision, Recall, Density, and Coverage. All can by done by modifying and running 
```
bash scripts/utils/prepare_data_imagenet_[res].sh
```
where [res] is substituted with the desired resolution (options are 64, 128, or 256). These scripts will assume that ImageNet is in a folder called `data` in the instance_selection_for_gans directory. Replace this with the filepath to your copy of ImageNet. 

#### 64x64 ImageNet
To replicate our best 64x64 model run `bash scripts/launch_SAGAN_res64_ch32_bs128_dstep_1_rr40.sh`. A single GPU with at least 12GB of memory should be sufficient to train this model. Training is expected to take about 2-3 days on a high-end GPU. 

We added only two configuration: 
```
parser.add_argument(
  '--maximum_focusing_rate', type=float, default=1,
  help='The percentage of maximum focusing rate (default: %(default)s)')

parser.add_argument(
  '--Training_type', type=str, default='without_SFL',
  choices=['without_SFL', 'SFL', 'SFL+'],
  help='Training type of SFL (default: %(default)s)')
    ```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Conditional Image Generation on ImageNet 64x64](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         |   IS &#8593;  |   FID &#8595; |  P &#8593; |  R &#8593; |  D &#8593; |  C &#8593; |
| ------------------ |--------| ------ |-----|-----|-----|-----|
|       SA-GAN       |  17.77 |  17.23 | 0.68| 0.66| 0.72| 0.71|
|      Approx SFL    |  19.11 |  16.20 | 0.69| 0.67| 0.76| 0.76|
|      Approx SFL+   |  21.50 |  14.20 | 0.72| **0.68**| 0.84| 0.80|
|      Exact SFL+    |  **21.98** |  **13.55** | **0.73**| 0.66| **0.85**| **0.81**|

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.
> 
## Applying Selective Focusing Learing to Your Own Dataset or Any Class-cGAN Network

Selective Focusing Learing can be applied to any class labelled PyTorch dataset using the `SFL` function. When classes are provided, Selective Focusing Learing will be applied separately for each class.  

The main hyperparameter for Selective Focusing Learing is the `retention_ratio`, a value from 0 to 100 which indicates the percentage of distribution matching that should be selected from the conditional matching. 

```python

  def SFL(self, out_c, out_u, Focusing_rate):
    out_c, idx_c = torch.sort(out_c, dim=0)
    out_u = out_u[idx_c[:, 0]]
    out = torch.cat([out_c[Focusing_rate:] + out_u[Focusing_rate:], out_c[:Focusing_rate]], 0)
    return out

  def SFL_plus(self, out_c, out_u, Focusing_rate, scores):
    _,idx_c = torch.sort(scores, dim=0, descending=True)
    out_c = out_c[idx_c]
    out_u = out_u[idx_c]
    out = torch.cat([out_c[Focusing_rate:] + out_u[Focusing_rate:], out_c[:Focusing_rate]], 0)
    return out
```

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.
