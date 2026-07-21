# Attribute Inference

Attribute inference attacks predict sensitive features of a data point (like gender or race) based on its outputs from a model, even when those features were not used as classification targets during training.

## Attack

To run an attribute inference attack, use `amulet.attribute_inference.attacks.DudduCIKM2022`. This attack trains an MLP on target model predictions and corresponding sensitive attributes to learn how to infer those attributes for new samples.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from amulet.attribute_inference.attacks import DudduCIKM2022
from amulet.attribute_inference.metrics import evaluate_attribute_inference
from amulet.utils import load_data, initialize_model, train_classifier

# 1. Load data with sensitive attributes
data = load_data("./data", "celeba")
if data.z_train is None:
    raise Exception("Dataset does not have sensitive attributes.")

# 2. Split data for adversary
split = train_test_split(data.x_train, data.y_train, data.z_train, test_size=0.5)
(x_target, x_adv, y_target, _, _, z_adv) = split

# 3. Train Target Model
target_model = initialize_model("vgg", "m1", data.num_features, data.num_classes).to(device)
target_model = train_classifier(target_model, target_train_loader, criterion, optimizer, 10, device)

# 4. Run Attribute Inference Attack
attribute_inference = DudduCIKM2022(
    target_model=target_model,
    x_train_adv=x_adv,
    x_test=data.x_test,
    z_train_adv=z_adv,
    batch_size=256,
    device=device
)

# results is a dict mapping attribute index to {predictions, confidence_values}
results = attribute_inference.attack()

# 5. Evaluate Metrics
metrics = evaluate_attribute_inference(data.z_test, results)
print(f"Attribute Inference Attack Accuracy: {metrics[0]['attack_accuracy']}")
```

## Metrics

Attribute inference is evaluated using **Balanced Accuracy** and **AUC Score** for each inferred attribute. Use `amulet.attribute_inference.metrics.evaluate_attribute_inference` to calculate these metrics from attack results.
