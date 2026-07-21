# Distribution Inference

Distribution inference attacks aim to determine which of two training distributions a model was trained on, for example whether a chosen sensitive attribute (like `sex` or `race`) appears in a given proportion of the training set. Amulet ships two attacks: `SuriEvans2022`, a black-box KL-divergence distinguishing test, and `WhiteBoxPIM`, a white-box attack that reads a model's raw parameters with a permutation-invariant meta-classifier.

Both attacks share the `DistributionInferenceAttack` lifecycle:

1. Construct the attack with the dataset arrays and a training configuration.
2. Call `prepare_model_populations()` to train (or load cached) the four model populations: adversary distribution 1, adversary distribution 2, victim distribution 1, and victim distribution 2.
3. Call `attack()` to run the distinguishing test.

## Black-Box Attack (KL divergence)

To run the KL-divergence distinguishing test, use `amulet.distribution_inference.attacks.SuriEvans2022`. It measures how each victim model's outputs diverge from the adversary baseline models trained on each distribution.

```python
from amulet.distribution_inference.attacks import SuriEvans2022
from amulet.distribution_inference.metrics import evaluate_distinguishing_accuracy
from amulet.utils import load_data

# 1. Load a dataset that exposes sensitive-attribute arrays (census or lfw)
data = load_data("./data", "census")

# 2. Construct the attack with the data arrays and a training config.
#    ratio1 and ratio2 are the two proportions of `filter_column == filter_value`
#    that define the distributions to distinguish.
dist_inf = SuriEvans2022(
    x_train=data.x_train,
    y_train=data.y_train,
    z_train=data.z_train,
    x_test=data.x_test,
    y_test=data.y_test,
    z_test=data.z_test,
    sensitive_columns=data.sensitive_columns,
    filter_column="sex",
    ratio1=0.1,
    ratio2=0.9,
    model_arch="linearnet",
    model_capacity="m1",
    num_features=data.num_features,
    num_classes=data.num_classes,
    num_models=5,       # Models trained per population
    epochs=1,
    batch_size=256,
    device=device,
    models_dir="./models/distribution_inference",
    dataset="census",
    exp_id=0
)

# 3. Train (or load) the four model populations, then run the attack.
dist_inf.prepare_model_populations()
results = dist_inf.attack()

# results["predictions"]: attack scores in [0, 1] (threshold at 0.5 to decide)
# results["ground_truth"]: 0 for distribution-1 victims, 1 for distribution-2 victims

# 4. Evaluate results
metrics = evaluate_distinguishing_accuracy(results["predictions"], results["ground_truth"])
print(f"Distinguishing Accuracy: {metrics['distinguishing_accuracy']}")
print(f"AUC Score: {metrics['auc_score']}")
```

The populations and test loaders built by `prepare_model_populations()` are the defaults for `attack()`. To score externally produced populations instead, pass them as keyword arguments (`models_adv_1`, `models_vic_1`, `test_loader_1`, and so on) to `attack()`.

## White-Box Attack (Permutation Invariant Model)

To run the white-box attack, use `amulet.distribution_inference.attacks.WhiteBoxPIM`. Rather than observing outputs, it reads each model's raw Linear and Conv2d weights and trains a Permutation Invariant Model (PIM) meta-classifier on the adversary populations to distinguish the two distributions, then evaluates that meta-classifier on the held-out victim populations.

`WhiteBoxPIM` takes the same constructor arguments as `SuriEvans2022`, plus meta-classifier settings (`meta_epochs`, `lr`, `inside_dims`).

```python
from amulet.distribution_inference.attacks import WhiteBoxPIM
from amulet.distribution_inference.metrics import evaluate_distinguishing_accuracy

whitebox = WhiteBoxPIM(
    x_train=data.x_train,
    y_train=data.y_train,
    z_train=data.z_train,
    x_test=data.x_test,
    y_test=data.y_test,
    z_test=data.z_test,
    sensitive_columns=data.sensitive_columns,
    filter_column="sex",
    ratio1=0.1,
    ratio2=0.9,
    model_arch="linearnet",
    model_capacity="m1",
    num_features=data.num_features,
    num_classes=data.num_classes,
    num_models=5,
    epochs=1,
    batch_size=256,
    device=device,
    models_dir="./models/distribution_inference",
    dataset="census",
    exp_id=0,
    meta_epochs=50,  # PIM meta-classifier training epochs
    lr=1e-2
)

whitebox.prepare_model_populations()
results = whitebox.attack()

metrics = evaluate_distinguishing_accuracy(results["predictions"], results["ground_truth"])
print(f"Distinguishing Accuracy: {metrics['distinguishing_accuracy']}")
```

## Metrics

Distribution inference is evaluated by the attack's **Distinguishing Accuracy**: how often the adversary correctly identifies which of the two distributions trained a victim model. Use `amulet.distribution_inference.metrics.evaluate_distinguishing_accuracy`. It takes `results["predictions"]` and `results["ground_truth"]` and returns a dict with keys `distinguishing_accuracy` and `auc_score`.
