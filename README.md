
>📋  A template README.md for code accompanying a Machine Learning paper

# Selective Focusing Learning for Conditional GANs

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

![distribution_overview1](https://user-images.githubusercontent.com/36159663/120271077-9d763080-c2e5-11eb-90cd-167ae185f0bc.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training BigGAN on ImageNet with Selective Focusing Learing

To train BigGAN models we use the [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) repo. We perform minimal changes to the code, consisting only of adding options for instance selection and additional metric logging. A list of changes made to the original repo can be found in the change log at [BigGAN-PyTorch/change_log.md](https://github.com/uoguelph-mlrg/instance_selection_for_gans/blob/master/BigGAN-PyTorch/change_log.md). 

#### Preparing Data
To train a BigGAN on ImageNet you will first need to construct an HDF5 dataset file for ImageNet (optional), compute Inception moments for calculating FID, and construct the image manifold for calculating Precision, Recall, Density, and Coverage. All can by done by modifying and running 
```
bash scripts/utils/prepare_data_imagenet_[res].sh
```
where [res] is substituted with the desired resolution (options are 64, 128, or 256). These scripts will assume that ImageNet is in a folder called `data` in the instance_selection_for_gans directory. Replace this with the filepath to your copy of ImageNet. 

#### 64x64 ImageNet
To replicate our best 64x64 model run `bash scripts/launch_SAGAN_res64_ch32_bs128_dstep_1_rr40.sh`. A single GPU with at least 12GB of memory should be sufficient to train this model. Training is expected to take about 2-3 days on a high-end GPU.

#### 128x128 ImageNet
To replicate 128x128 ImageNet results run `bash scripts/launch_BigGAN_res128_ch64_bs256_dstep_1_rr50.sh`. This script assumes that training will be done on 8 GPUs with 16GB of memory each. To train with less GPUs, or if you encounter out-of-memory errors, you can try reducing `batch_size` and increasing `num_G_accumulations` and `num_D_accumulations` to achieve the desired effective batch size (effective batch size = batch_size x num_accumulations).
To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.
