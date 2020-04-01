# Resources for releasing research code in Machine Learning

This repository contains resources for those wanting to publish a high impact ML research repository. 

## README.md template

We provide a [README.md template](templates/README.md) for releasing ML research repositories.  

## ML Code Completeness Checklist

Goals of the ML Code Completeness Checklist:
- facilitate reproducibility
- make it easier for others build upon research code 

We find that repositories that follow all of the checklist items get many more github stars (on median 200 more, and on average 2500 more). 

For more details behind the checklist items please refer to our [medium post](https://medium.com/paperswithcode/). We also provide the [data](notebooks/code_checklist-neurips2019.csv) and [notebook](notebooks/code_checklist-analysis.Rmd) to reproduce this analysis from the medium post. 

The checklist is made to be as general as possible. It consists of 5 items:

1. Specification of dependencies
2. Training code
3. Evaluation code
4. Table/Figure of the main result, with instruction to reproduce it
5. Pre-trained model

We'll explain each one in some detail below. 

#### 1. Specification of dependencies

If you are using Python, this means providing a `requirements.txt` file (if using `pip` and `virtualenv`), providing `environment.yml` file (if using anaconda), or a `setup.py` if your code is a library. 

It is good practice to provide a section in your README.md that explains how to install these dependencies. Assume minimal background knowledge and be clear and comprehensive - if users cannot set up your dependencies they are likely to give up on the rest of your code as well. 

If you wish to provide whole reproducible environments, you might want to consider using Docker and upload a Docker image of your environment into Dockerhub. 

#### 2. Training code

Training models, even when the whole model is well-specific, can be very difficult. Include scripts that mirror your training setup as closely as possible, so users can understand how you trained your models and can re-use your methodology on their own dataset. 

This should include hyperparameters and training tricks you are using in your training. 

You can also provide a documented command line wrapper such as `train.py` to serve as a useful entry point for your users. 

#### 3. Evaluation code

Evaluating models and running experiments is frequently filled with subtle details that are not always possible to explain in the paper. If you include the exact code you used to evaluate your models or run different comparison scenarios, this can be useful to the user to understand and build on your research. 

You can also provide a documented command line wrapper such as `eval.py` to serve as a useful entry point for your users interested in understanding how you evaluated your models. 

#### 4. Table/Figure of the main result, with instruction to reproduce it

Adding a table/figure of results into README.md lets your users quickly understand what to expect from the repository (see the [README.md template](templates/README.md) for an example). Instructions on how to reproduce those results (with links to any relevant scripts, pretrained models etc) can provide another entry point for the user and directly facilitate reproducibility.  

Linking back to the full leaderboard enables user to put your results in context and enhance trust in those results. 

#### 5. Pre-trained models

Training the model from scratch might take a long time, so releasing the model lets the community evaluate it without having to train it themselves. 

Releasing models is also useful if it's quick to train, as it serves as a point of comparison. Finally, even if you don't expect people to use your model for fine-tuning or any other downstream task, it is still useful to release it so the community can try it out.

## Awesome resources for releasing research code

### Hosting pretrained models files

1. [GitHub Releases](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository) - versioning, 2GB file limit, free bandwidth
2. [Google Drive](https://drive.google.com) - versioning, 5TB, free bandwidth
3. [Dropbox](https://dropbox.com) - versioning, 2GB (paid unlimited), free bandwidth
4. [AWS S3](https://aws.amazon.com/s3/) - versioning, paid only, paid bandwidth
 
### Managing model files

1. [RClone](https://rclone.org/) - provides unified access to many different cloud storage providers

### Standardized model interfaces

1. [PyTorch Hub](https://pytorch.org/hub/)
2. [Tensorflow Hub](https://www.tensorflow.org/hub)
3. [Hugging Face NLP models](https://huggingface.co/models)

### Results leaderboards

1. [Papers with Code leaderboards](https://paperswithcode.com/sota)
2. [CodaLab](https://competitions.codalab.org/)
3. [EvalAI](https://evalai.cloudcv.org/)

### Making project pages

1. [GitHub pages](https://pages.github.com/)
2. [Fastpages](https://github.com/fastai/fastpages)

### Making demos and tutorials

1. [Google Colab](https://colab.research.google.com/)
2. [Binder](https://mybinder.org/)
3. [Streamlit](https://github.com/streamlit/streamlit)
