>📋  A template README.md for code accompanying a Machine Learning paper

# Meta Compression: Learning to compress Deep Neural Networks

This repository is the official implementation of Meta Compression: Learning to compress Deep Neural Networks



**Abstract.** _Deploying large pretrained deep learning models is hindered by the limitations of realistic scenarios such as resource constraints on the user/edge devices. Issues such as selecting the right pretrained model, compression method, and compression level to suit a target application and hardware become especially important. We address these challenges using a novel meta learning framework that can provide high quality recommendations tailored to the specified resource, performance, and efficiency constraints._

_For scenarios with limited to no access to unseen samples that resemble the distribution used for pretraining, we invoke diffusion models to improve generalization to test data and thereby demonstrate the promise of augmenting meta-learners with generative models. When learning across several state-of-the-art compression algorithms and DNN architectures trained on the CIFAR10 dataset, our top recommendation shows only 1% drop in average accuracy loss compared to the optimal compression method. This is in contrast to 25% average accuracy drop achieved by selecting the single best compression method across all constraints._


>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

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
