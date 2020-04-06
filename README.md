# Research Code Guidelines

**This repository contains the official guidelines for NeurIPS 2020 code submission**. 

These guidelines assist researchers in maximizing the impact of their research with high-quality code repositories.

For NeurIPS 2020 code submissions we recommend using a [README.md template](#readmemd-template) and checking as many items on the [ML Code Completeness Checklist](#ml-code-completeness-checklist) as possible. 

All recommendations are based on a [data-driven analysis](https://medium.com/paperswithcode) of what makes research code repositories have high impact. 

## README.md template

We provide a [README.md template](templates/README.md) that you can use for releasing ML research repositories. The sections in the template were derived by looking at existing repositories, seeing which had the best reception in the community, and then looking at common components that correlate with perceived usefulness (see next Section).

## ML Code Completeness Checklist

The goals of the ML Code Completeness Checklist are to:
- facilitate reproducibility
- make it easier for others build upon research code 

We find that those repositories that score higher on the checklist tend to have a higher number of GitHub stars. 

We've verified this by analysing NeurIPS 2019 repositories. For more details on this analyis please refer to our [medium post](https://medium.com/paperswithcode/). We also provide the [data](notebooks/code_checklist-neurips2019.csv) and [notebook](notebooks/code_checklist-analysis.Rmd) to reproduce this analysis from the post. 

The checklist is made to be as general as possible. It consists of five items:

1. Specification of dependencies
2. Training code
3. Evaluation code
4. Table/Figure of the main result, with instruction to reproduce it
5. Pre-trained model

We found that NeurIPS 2019 repositories that have all five of these components got the highest number of GitHub stars. 

We'll explain each one in some detail below. 

#### 1. Specification of dependencies

If you are using Python, this means providing a `requirements.txt` file (if using `pip` and `virtualenv`), providing `environment.yml` file (if using anaconda), or a `setup.py` if your code is a library. 

It is good practice to provide a section in your README.md that explains how to install these dependencies. Assume minimal background knowledge and be clear and comprehensive - if users cannot set up your dependencies they are likely to give up on the rest of your code as well. 

If you wish to provide whole reproducible environments, you might want to consider using Docker and upload a Docker image of your environment into Dockerhub. 

#### 2. Training code

Your code should have a training script that can be used to obtain the principal results stated in the paper. This means you should include hyperparameters and any tricks that were used in the process of getting your results. To maximize usefulness, ideally this code should be written with extensibility in mind: what if your user wants to use the same training script on their own dataset?

You can provide a documented command line wrapper such as `train.py` to serve as a useful entry point for your users. 

#### 3. Evaluation code

Model evaluation and experiments often depend on subtle details that are not always possible to explain in the paper. This is why including the exact code you used to evaluate or run experiments is helpful to give a complete description of the procedure. In turn, this helps the user to trust, understand and build on your research.

You can provide a documented command line wrapper such as `eval.py` to serve as a useful entry point for your users.

#### 4. Table/Figure of the main result, with instruction to reproduce it

Adding a table/figure of results into README.md lets your users quickly understand what to expect from the repository (see the [README.md template](templates/README.md) for an example). Instructions on how to reproduce those results (with links to any relevant scripts, pretrained models etc) can provide another entry point for the user and directly facilitate reproducibility. 

You can also put your results in context by linking back to a full leaderboard of results on the same benchmark, aiding comparison with a larger set of methodologies.

#### 5. Pre-trained models

Training a model from scratch can be time-consuming and expensive. One way to increase trust in your results is to provide a pre-trained model that the community can evaluate to obtain the end results. This means users can see the results are credible without having to train afresh.

Another common use case is fine-tuning for downstream task, where it's useful to release a pretrained model so others can build on it for application to their own datasets.

Lastly, some users might want to try out your model to see if it works on some example data. Providing pre-trained models allows your users to play around with your work and aids understanding of the paper's achievements.

## Other awesome resources for releasing research code

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
3. [NLP Progress](https://nlpprogress.com/)
4. [EvalAI](https://evalai.cloudcv.org/)

### Making project pages

1. [GitHub pages](https://pages.github.com/)
2. [Fastpages](https://github.com/fastai/fastpages)

### Making demos and tutorials

1. [Google Colab](https://colab.research.google.com/)
2. [Binder](https://mybinder.org/)
3. [Streamlit](https://github.com/streamlit/streamlit)
