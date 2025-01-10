# How to contribute to Amulet

## Reporting an Issue

We use the Issue tracking feature to track bugs and new code contributions.

### Creating a bug report

Each issue should include a title and a description of the error you are facing. Ensure to include as much relevant information as possible, including a code sample or failing test demonstrating the expected behavior and your system configuration. Your goal should be to make it easy for yourself - and others - to reproduce the bug and figure out a fix.

### Feature Requests

Since this is a growing package, we welcome new feature requests! However, remember that you may need to write the code for a feature yourself. Depending on the type of feature you want, there are slightly different requirements. Some examples are:
- **Requesting a utility for an ML Pipeline.**
If this is an easy fix and we feel this would be helpful to many users facing the same issue, we would love to work with you on this to make it happen!
- **Adding a new risk or defense.**
Are you a researcher who has discovered a new risk or way to defend against known risks? We welcome your contributions! However, in most cases, we only include state-of-the-art risks or defenses in our package. The package aims to allow other users to test their models against known risks or defenses or enable researchers to test their techniques against the current state-of-the-art. Thus, having a peer-reviewed paper to justify adding a new risk or defense would be nice.

## Contributing Code

For new functionalities that help with an ML pipeline, please submit an issue, and we can work together to find the best way to incorporate the utility. For a module comprising a new risk or defense, we strongly urge you to follow the same coding conventions as the rest of the package. Please follow the tutorial below to add a new module.

**For all contributions:**
- **Submit an issue describing the contribution**.
- **Fork or clone the library.**
- **Create a new branch for your contribution.**
- **After adding the code, create a merge request.**

### Setting up your environment

#### Install poetry

> [!CAUTION]
> This repo doesn't work with `poetry` 2.0. Use an earlier version, e.g. `1.8.5`.

`python3 -m venv .poetry_venv`

`. .poetry_venv/bin/activate` or `. .venv/bin/activate.fish`

`python -m pip install --upgrade pip`

`pip install poetry==1.8.5`

`deactivate`

Consider setting `.poetry_venv/bin/poetry config virtualenvs.create false` to prevent poetry from creating its own venv.

#### Main venv

To create the virtual environemnt:
`python3 -m venv .venv`

To activate it:
`source .venv/bin/activate` or if using fish `. .venv/bin/activate.fish`

Then, to install the dependencies:
`.poetry_venv/bin/poetry install`

**DISCLAIMER:** Installing `pytorch` with `poetry` is [still weird](https://github.com/python-poetry/poetry/blob/main/docs/repositories.md#explicit-package-sources) but should work.

#### Using poetry

(Inside your `.venv`);
when you add or modify any dependencies in `pyproject.toml`, run `.poetry_venv/bin/poetry lock --no-update` to rebuild the dependency graph.
Then run `.poetry_venv/bin/poetry install` to install the dependencies.

#### pre-commit

There're some pre-commit hooks configured for this project.
Also, `poetry` installs `pre-commit` as a dev dependency.

Run `pre-commit install` for consistent development.

## Additional Features

### Adding a dataset

By default, dataset loading functions in Amulet return train and test sets as [`torch.utils.data.TensorDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset) objects. For datasets with sensitive attributes, users can set a flag that returns NumPy arrays instead. We recommend you look at the existing implementations as an example, specifically [`amulet/utils/_pipeline.py:load_data()`](https://github.com/ssg-research/amulet/blob/main/amulet/utils/_pipeline.py#L16) and [`amulet/datasets/_image_datasets.py`](https://github.com/ssg-research/amulet/blob/main/amulet/datasets/_image_datasets.py). The function template looks like this:

```python
def load_<name_of_dataset> (
	path: Union[str, Path],  # indicating where to store the dataset once downloaded,
	random_seed: Optional[int], # used if the function splits the dataset into train/test
	return_x_y: Optional[boolean] # flag used to return NumPy arrays, if applicable
) -> sklearn.utils.Bunch
```
Note that the output is a [`sklearn.utils.Bunch`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html) object, a dictionary-like object of the format:
```python
{
	train_set: torch.utils.data.TensorDataset,
	test_set: torch.utils.data.TensorDataset
}
```
Please follow these steps to add a new function to load a dataset:
1. Write a function that will download the dataset, preprocess it, and split it into train and test sets (see function template above). Example code: [`amulet/datasets/_tabular_datasets.py:load_census()`](https://github.com/ssg-research/amulet/blob/main/amulet/datasets/_tabular_datasets.py#L21). Please ensure the following when:
    * **Downloading**: Ensure download location is passed as a parameter. Example code: [`amulet/datasets/_tabular_datasets.py:L72-81`](https://github.com/ssg-research/amulet/blob/main/amulet/datasets/_tabular_datasets.py#L72-L81).
    * **Preprocessing**: Ensure the data types are correct, engineer features, etc. Example code: [`amulet/datasets/_tabular_datasets.py:L83-91`](https://github.com/ssg-research/amulet/blob/main/amulet/datasets/_tabular_datasets.py#L83-L91).
    * **Splitting**: Ensure any randomized splitting uses a random seed. Example code: [`amulet/datasets/_tabular_datasets.py:L94-99`](https://github.com/ssg-research/amulet/blob/main/amulet/datasets/_tabular_datasets.py#L94-L99).
2. Add this function into the appropriate file: `amulet/datasets/_tabular_datasets.py` for 1-dimensional datasets and `amulet/datasets/_image_datasets.py` for 2-dimensional datasets.
3. Import your function in [`amulet/datasets/__init__.py`](https://github.com/ssg-research/amulet/blob/main/amulet/datasets/__init__.py).
4. Add the appropriate if condition in [`amulet/utils/_pipeline.py/load_data()`](https://github.com/ssg-research/amulet/blob/main/amulet/utils/_pipeline.py).

### Adding a model architecture

Please follow these steps to add a new model architecture to Amulet:
1. Create a file in `amulet/models/` that defines a model as a `nn.Module` subclass.
2. We recommend including a `get_hidden()` function in the model since some modules use it. This function outputs the model's hidden layer activations. Example code: [`amulet/models/vgg.py:L65-76`](https://github.com/ssg-research/amulet/blob/main/amulet/models/vgg.py#L65-L76).
3. Import the new model into `amulet/models/__init__.py`. For example, `from model_file import model_name`.
4. We also recommend configuring the model size and complexity via input parameters. Please see [`amulet/models/vgg.py`](https://github.com/ssg-research/amulet/blob/main/amulet/models/vgg.py) or [`amulet/models/binary_net.py`](https://github.com/ssg-research/amulet/blob/main/amulet/models/binary_net.py) for examples.

## Adding a new module

The first step is to decide which risk the new module interacts with. For details, refer to the Tutorial (link TBD). The risks currently defined by Amulet can be found in this table (link TBD). Then, decide whether the module is a metric, attack, or defense. If you have identified a new risk, you may need to add separate modules for a metric, attack and/or defense.

### Adding an attack or a defense

The general template of an attack or defense:
- Inputs:
    - Target Model
    - Hyperparameters (include reasonable default values)
    - Other attributes or methods required by the class
- Main Algorithm:
    - Code logic to run the attack or defense
- Outputs:
    - Output of the attack OR
    - Defended model

Please refer to the Module Templates (link TBD) for an idea of the outputs for our existing modules. This ensures that new attacks or defenses can be compared to old ones using the same code. **If introducing a new metric, please make two separate contributions for the metric and the new attack or defense.**

Please note that your attack or defense **should not output a metric**. Instead, it should output the data that will be input to one of the existing metrics implemented in the pipeline.

### Metrics

There is no set template for metrics. Please include in-code documentation about the expected input for the metric calculation and the output range.

### Steps

Once a rough template has been sketched out, please follow these steps for adding code:
1. Create a file in the appropriate directory as follows:

    `amulet/<risk>/<defense/attack/metric>/new_module.py`

    For example, if your module is a new defense for evasion, create the file as follows:

    `amulet/evasion/defense/new_module.py`
2. Create a class. We follow a convention for all our attacks and defenses:
    1. Create a subclass using the Base class for the attack/defense you want to add. For example, to add a new evasion attack, you can use the following code:
        ```python

        from .evasion import Evasion
        class newEvasionAttack(Evasion):
            def __init__(
                model,
                test_loader,
                device,
                batch_size,
                param1,
                param2
            ):
                super().__init__(model, test_loader, device, batch_size)
                self.param1 = param1
                self.param2 = param2
        ```
    2. The `__init__()` method should have most of the parameters required to run the technique. This allows users to use functions like `__getattr__()` to log the parameters while running the technique. Optional parameters may go in other methods.
    3. Where possible, use the utils available in the package.
    4. If your module uses hyperparameters, please recommend a reasonable default value.
    5. Add docstrings to the class and methods, including the recommended range for specific hyperparameters for your technique. Please use the modules we have written as a reference for formatting these.
    6. We follow the convention of prefixing the method with an underscore for private attributes and methods in the class.
3. Please include the code to download any files or data the module requires.
4. Add an example script in `<TBD>` to show how to use your technique. Please look at the existing examples to understand how to use them.
5. Run PyLint on your code using the pylintrc file in the repo to validate the code style.
6. If the technique requires additional packages, include them in the environment.yml file with the appropriate version number.
