# Unauthorized Model Ownership

Unauthorized Model Ownership risks are related to an adversary being able to "steal" a model, such that the stolen (surrogate) model has the same behavior and characteristics as the target model. Amulet provides tools for **Model Extraction** attacks and defenses based on **Watermarking** and **Fingerprinting**.

## Model Extraction Attack

To run a model extraction attack, use `amulet.unauth_model_ownership.attacks.ModelExtraction`. This attack trains a surrogate (attack) model by querying the target model.

```python
from amulet.unauth_model_ownership.attacks import ModelExtraction
from amulet.utils import initialize_model, load_data

# Load data and initialize target/attack models
data = load_data("./data", "cifar10")
target_model = initialize_model("vgg", "m1", data.num_features, data.num_classes)
attack_model = initialize_model("vgg", "m1", data.num_features, data.num_classes).to(device)

train_loader = DataLoader(data.train_set, batch_size=256)
optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-3)

# Configure and run the attack
model_extraction = ModelExtraction(
    target_model=target_model,
    attack_model=attack_model,
    optimizer=optimizer,
    train_loader=train_loader,
    device=device,
    epochs=50,
    loss_type="mse" # Alternatives: "kl", "ce"
)

stolen_model = model_extraction.attack()
```

## Fingerprinting Defense

Fingerprinting determines if a suspect model was stolen by measuring how its outputs diverge on sensitive data points. Amulet implements **Dataset Inference** as a fingerprinting mechanism.

To run fingerprinting, use `amulet.unauth_model_ownership.defenses.DatasetInference`.

```python
from amulet.unauth_model_ownership.defenses import DatasetInference

# Initialize Fingerprinting mechanism
fingerprinting = DatasetInference(
    target_model=target_model,
    suspect_model=stolen_model,
    train_loader=train_loader,
    test_loader=test_loader,
    num_classes=data.num_classes,
    device=device,
    dataset="2D" # Use "1D" for tabular data
)

# Run Fingerprinting and get p-values for membership
results = fingerprinting.fingerprint()

print(f"Suspect model p-value: {results['suspect']['p-value']}")
if results['suspect']['p-value'] < 0.05:
    print("Suspect model is likely stolen.")
```

## Watermarking Defense

Watermarking involves embedding a "secret" behavior (backdoor) into the model during training. Amulet implements the **WatermarkNN** algorithm.

To watermark a model, use `amulet.unauth_model_ownership.defenses.WatermarkNN`.

```python
from amulet.unauth_model_ownership.defenses import WatermarkNN

# Configure and apply Watermarking
wm_model_wrapper = WatermarkNN(
    model=target_model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader,
    device=device,
    wm_path="./miscellaneous/trigger_set/",
    epochs=50
)

watermarked_model = wm_model_wrapper.watermark()
```

## Metrics

### Model Extraction Metrics

Use `amulet.unauth_model_ownership.metrics.evaluate_extraction` to evaluate surrogate models:

```python
from amulet.unauth_model_ownership.metrics import evaluate_extraction

results = evaluate_extraction(target_model, stolen_model, test_loader, device)
print(f"Fidelity (model agreement): {results['fidelity']}%")
```

### Fingerprinting / Watermarking

Fingerprinting results are primarily interpreted via **p-values** (statistical significance). Watermarking success is measured by the model's accuracy on the trigger set while maintaining performance on clean data.
