# Amulet
Amulet is a Python machine learning (ML) package to evaluate the susceptibility of different risks to security, privacy, and fairness. Amulet is applicable to evaluate how algorithms designed to reduce one risk may impact another unrelated risk and compare different attacks/defenses for a given risk. 

Amulet builds upon prior work titled [“SoK: Unintended Interactions among Machine Learning Defenses and Risks”](https://arxiv.org/abs/2312.04542) which appears in IEEE Symposium on Security and Privacy 2024. The SoK covers only two interactions and identifies the design of a software library to evaluate unintended interactions as future work. Amulet addresses this gap by including eight different risks each covering their own attacks, defenses and metrics. 

Amulet is:
- Comprehensive: Covers the most representative attacks/defenses/metrics for different risks.
- Extensible: Easy to include additional risks, attacks, defenses, or metrics.
- Consistent: Allows using different attacks/defenses/metrics with a consistent, easy-to-use API.
- Applicable: Allows evaluating unintended interactions among defenses and attacks.


Built to work with PyTorch, you can incorporate Amulet into your current ML pipeline to test how your model interacts with these state-of-the-art defenses and risks. Alternatively, you can use the example pipelines to bootstrap your pipeline.


## Requirements
We recommend using **conda**. Create a virtual environment and install requirements:

```bash
conda env create -f environment.yml
```

To activate:

```bash
conda activate amulet
```

**Note:** The package requires the CUDA version to be 11.8 or above for PyTorch 2.2

## Features
We provide a high-level list of features below. Please refer to the Tutorial for more information.

### Datasets
Amulet provides the following for computer vision tasks:
- [CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html).
- [FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).

Amulet pre-processes and provides the following datasets to test for privacy-related concerns:
- [Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income).
- [Labeled Faces in the Wild (LFW)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html).

### Models
Amulet provides the following models:
- [VGG](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/): for computer vision tasks.
- LinearNet: A dense neural network tuned for multiclass classification on the FashionMNIST dataset.
- BinaryNet: A dense neural network tuned for binary classification.

All models have configurable sizes to evaluate the impact of model capacity. Please see [Defining a Model]() for more details.

### Risks
Amulet provides attacks, defenses, and evaluation metrics for the following risks:
#### Security
- Evasion
- Poisoning
- Unauthorized Model Ownership

#### Privacy
- Membership Inference
- Attribute Inference
- Distribution Inference
- Data Reconstruction

#### Fairness
- Discriminatory Behavior

Please check the Module Heirarchy for more details.

