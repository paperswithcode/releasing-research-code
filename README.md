> 🧩This is a template for code README.md accompanying a Machine Learning paper

# My owesome paper title

This repository is the official implementation of [My owesome paper title](https://arxiv.org/abs/2030.12345). 

> 🧩Optional: include a graphic explaining your approach or main result. 

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

> 🧩Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model in the paper, run this command:

```
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

> 🧩Describe how to train the model, with example commands on how to train the models in your paper, including the full training procedure and hyperparameter optimisation approach.

## Evaluation

To evaluate my model on ImageNet, run:

```
python eval.py --model-file mymodel.pth --benchmark imagenet
```

> 🧩Describe how to evaluate the trained models on benchmarks reported in the paper, give example commands. 

## Pre-trained models

We provide links to pretrained models:

- [mymodel.pth](https://drive.google.com/filehash)

> 🧩Give a link to where/how the pretrained models can be downloaded and used (if applicable). 

## Results

Our model achieves the following performance on :

- [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name      | Top 1 Accuracy  | Top 5 Accuracy |
| --------------- |---------------- | -------------- |
| My model        |     85%         |      95%       |

> 🧩Include a table of results from your paper, and link back to the leaderboard to give readers more context in the future. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


