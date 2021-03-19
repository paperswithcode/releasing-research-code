# Tips for Publishing Research Code

<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" width=200>

**💡 Collated best practices from most popular ML research repositories - *now official guidelines at NeurIPS 2021!*** 

Based on analysis of more than 200 Machine Learning repositories, these recommendations facilitate reproducibility and correlate with GitHub stars - for more details, see our [our blog post](https://medium.com/paperswithcode/ml-code-completeness-checklist-e9127b168501). 

For NeurIPS 2021 code submissions it is recommended (but not mandatory) to use the [README.md template](templates/README.md) and check as many items on the ML Code Completeness Checklist (described below) as possible. 

## 📋 README.md template

We provide a [README.md template](templates/README.md) that you can use for releasing ML research repositories. The sections in the template were derived by looking at existing repositories, seeing which had the best reception in the community, and then looking at common components that correlate with popularity.

## ✓ ML Code Completeness Checklist

We compiled this checklist by looking at what's common to the most popular ML research repositories. In addition, we prioritized items that facilitate reproducibility and make it easier for others build upon research code.

The ML Code Completeness Checklist consists of five items:

1. **Specification of dependencies**
2. **Training code** 
3. **Evaluation code**
4. **Pre-trained models**
5. **README file including table of results accompanied by precise commands to run/produce those results**

We verified that repositories that check more items on the checklist also tend to have a higher number of GitHub stars. This was verified by analysing official NeurIPS 2019 repositories - more details in the [blog post](https://medium.com/paperswithcode/ml-code-completeness-checklist-e9127b168501). We also provide the [data](notebooks/code_checklist-neurips2019.csv) and [notebook](notebooks/code_checklist-analysis.pdf) to reproduce this analysis from the post. 

NeurIPS 2019 repositories that had all five of these components had the highest number of GitHub stars (median of 196 and mean of 2,664 stars). 

We explain each item on the checklist in detail blow. 

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

#### 4. Pre-trained models

Training a model from scratch can be time-consuming and expensive. One way to increase trust in your results is to provide a pre-trained model that the community can evaluate to obtain the end results. This means users can see the results are credible without having to train afresh.

Another common use case is fine-tuning for downstream task, where it's useful to release a pretrained model so others can build on it for application to their own datasets.

Lastly, some users might want to try out your model to see if it works on some example data. Providing pre-trained models allows your users to play around with your work and aids understanding of the paper's achievements.

#### 5. README file includes table of results accompanied by precise command to run to produce those results

Adding a table of results into README.md lets your users quickly understand what to expect from the repository (see the [README.md template](templates/README.md) for an example). Instructions on how to reproduce those results (with links to any relevant scripts, pretrained models etc) can provide another entry point for the user and directly facilitate reproducibility. In some cases, the main result of a paper is a Figure, but that might be more difficult for users to understand without reading the paper. 

You can further help the user understand and contextualize your results by linking back to the full leaderboard that has up-to-date results from other papers. There are [multiple leaderboard services](#results-leaderboards) where this information is stored.  

## 🎉 Additional awesome resources for releasing research code

### Hosting pretrained models files

1. [Zenodo](https://zenodo.org) - versioning, 50GB, free bandwidth, DOI, provides long-term preservation
2. [GitHub Releases](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository) - versioning, 2GB file limit, free bandwidth
3. [OneDrive](https://www.onedrive.com/) - versioning, 2GB (free)/ 1TB (with Office 365), free bandwidth
4. [Google Drive](https://drive.google.com) - versioning, 15GB, free bandwidth
5. [Dropbox](https://dropbox.com) - versioning, 2GB (paid unlimited), free bandwidth
6. [AWS S3](https://aws.amazon.com/s3/) - versioning, paid only, paid bandwidth
7. [huggingface_hub](https://github.com/huggingface/huggingface_hub) - versioning, no size limitations, free bandwidth
8. [DAGsHub](https://dagshub.com/) - versioning, no size limitations, free bandwith
9. [CodaLab Worksheets](https://worksheets.codalab.org/) - 10GB, free bandwith
 
### Managing model files

1. [RClone](https://rclone.org/) - provides unified access to many different cloud storage providers

### Standardized model interfaces

1. [PyTorch Hub](https://pytorch.org/hub/)
2. [Tensorflow Hub](https://www.tensorflow.org/hub)
3. [Hugging Face NLP models](https://huggingface.co/models)
4. [ONNX](https://onnx.ai/)

### Results leaderboards

1. [Papers with Code leaderboards](https://paperswithcode.com/sota) - with 4000+ leaderboards
2. [CodaLab Competitions](https://competitions.codalab.org/) - with 450+ leaderboards
3. [EvalAI](https://eval.ai/) - with 100+ leaderboards
4. [NLP Progress](https://nlpprogress.com/) - with 90+ leaderboards
5. [Collective Knowledge](https://cKnowledge.io/reproduced-results) - with 40+ leaderboards
6. [Weights & Biases - Benchmarks](https://www.wandb.com/benchmarks) - with 9+ leaderboards

### Making project pages

1. [GitHub pages](https://pages.github.com/)
2. [Fastpages](https://github.com/fastai/fastpages)

### Making demos, tutorials, executable papers

1. [Google Colab](https://colab.research.google.com/)
2. [Binder](https://mybinder.org/)
3. [Streamlit](https://github.com/streamlit/streamlit)
4. [CodaLab Worksheets](https://worksheets.codalab.org/)

## Contributing

If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at hello@paperswithcode.com or open an issue on this GitHub repository. 

All contributions welcome! All content in this repository is licensed under the MIT license.
