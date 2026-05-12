# Distribution Inference

Distribution inference attacks aim to determine if a training set follows a specific distribution (e.g., if a certain property or ratio of samples is present). Amulet implements a KL-divergence-based distinguishing test.

## Attack

To run a distribution inference attack, use `amulet.distribution_inference.attacks.SuriEvans2022`. This attack measures how a victim model's outputs diverge from adversary baseline models trained on two different distributions.

```python
from amulet.distribution_inference.attacks import SuriEvans2022

# 1. Prepare Victim and Adversary Models for both distributions
# victim_models_1: list[nn.Module], victim_models_2: list[nn.Module]
# adversary_models_1: list[nn.Module], adversary_models_2: list[nn.Module]

# 2. Run the Distribution Inference attack
dist_inf = SuriEvans2022(
    models_vic_1=victim_models_1,
    models_vic_2=victim_models_2,
    models_adv_1=adversary_models_1,
    models_adv_2=adversary_models_2,
    test_loader_1=test_loader_dist_1,
    test_loader_2=test_loader_dist_2,
    device=device
)

results = dist_inf.attack()

# 3. Evaluate results
# results["predictions"] contains scores in [0, 1]
# results["ground_truth"] contains binary labels for distributions
```

## Metrics

Distribution inference is evaluated by the attack's **Distinguishing Accuracy**. This measures how well the adversary can correctly identify which of the two distributions was used to train a victim model. Use `amulet.distribution_inference.metrics.DistinguishingAccuracy` to compute these metrics.
