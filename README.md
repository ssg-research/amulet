# ML-Conf
ML-Conf is a Python package for evaluating machine learning (ML) models against known risks and analyzing how known defenses interact with various risks.

Built to work with PyTorch, you can incorporate ML-Conf into your current ML pipeline to test how your model interacts with these state-of-the-art defenses and risks. Alternatively, you can use the example pipelines to bootstrap your pipeline.

## Requirements
We recommend using **conda**. Create a virtual environment and install requirements:

```bash
conda env create -f environment.yml
```

To activate:

```bash
conda activate mlconf
```

**Note:** The package requires the CUDA version to be 11.8 or above for PyTorch 2.2
## Usage


 
## Features
### Datasets
ML-Conf provides the following for computer vision tasks:
- [CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html).
- [FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).

ML-Conf processes and provides the following datasets to test for privacy-related concerns:
- [Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income).
- [Labeled Faces in the Wild (LFW)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html).

### Models
ML-Conf provides the following models:
- [VGG](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/): for computer vision tasks.
- LinearNet: A dense neural network tuned for multiclass classification on the FashionMNIST dataset.
- BinaryNet: A dense neural network tuned for binary classification.

All models have configurable sizes to evaluate the impact of model capacity. Please see [Defining a Model](#defining-a-model) for more details.

### Defenses
ML-Conf provides the following defenses that you can evaluate on your model:
- [Adversarial Training](https://github.com/cleverhans-lab/cleverhans)
- Outlier Removal

TO-DO:
- [DP-SGD](https://github.com/pytorch/opacus)
- [Group Fairness](https://xebia.com/blog/towards-fairness-in-ml-with-adversarial-networks/)
- [Fingerprinting](https://github.com/cleverhans-lab/dataset-inference)
- [Watermarking](https://github.com/adiyoss/WatermarkNN/tree/master)
- [Explanations](https://github.com/pytorch/captum)
 
### Risks 
 ML-Conf provides the following risks you can run against your model:
- [Evasion using Adversarial Examples](https://github.com/cleverhans-lab/cleverhans)
- [Model Extraction](https://github.com/liuyugeng/ML-Doctor)
- [Data Reconstruction](https://github.com/liuyugeng/ML-Doctor)
- Discriminatory Behavior

TO-DO:
- Membership Inference (https://github.com/YuxinWenRick/canary-in-a-coalmine) 
- Attribute Inference (https://github.com/vasishtduddu/AttInfExplanations)
- Poisoning (https://github.com/Billy1900/BadNet)
- Distribution Inference (https://github.com/iamgroot42/dissecting_dist_inf and https://github.com/iamgroot42/FormEstDistRisks)

