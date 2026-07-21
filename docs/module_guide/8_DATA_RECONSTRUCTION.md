# Data Reconstruction

Data reconstruction attacks (also known as model inversion) attempt to recover training data samples by exploiting a model's confidence outputs. Amulet implements the Fredrikson et al. attack.

## Attack

To run a data reconstruction attack, use `amulet.data_reconstruction.attacks.FredriksonCCS2015`. This attack uses gradient descent to reconstruct an "average" data point for each output class based on the model's confidence scores.

```python
import torch
from torch.utils.data import DataLoader
from amulet.data_reconstruction.attacks import FredriksonCCS2015
from amulet.data_reconstruction.metrics import evaluate_similarity
from amulet.utils import load_data, initialize_model, train_classifier

# 1. Load data and train target model
data = load_data("./data", "lfw")
target_model = initialize_model("vgg", "m1", data.num_features, data.num_classes).to(device)
target_model = train_classifier(target_model, train_loader, criterion, optimizer, 10, device)

# 2. Configure and run Data Reconstruction attack
input_size = (1,) + tuple(data.test_set[0][0].shape)
output_size = data.num_classes

# FredriksonCCS2015 needs a Softmax-output model: its cost is 1 - p_target, so the
# final layer must emit class probabilities. Wrap the classifier accordingly.
inversion_model = torch.nn.Sequential(target_model, torch.nn.Softmax(dim=1))

data_recon = FredriksonCCS2015(
    target_model=inversion_model,
    input_size=input_size,
    output_size=output_size,
    device=device,
    alpha=3000 # Number of iterations
)

# Returns a list of reconstructed tensors, one per class
reconstructed_data = data_recon.attack()

# 3. Evaluate Metrics. evaluate_similarity iterates the loader one sample at a time,
# so it must be built with batch_size=1.
test_loader = DataLoader(data.test_set, batch_size=1, shuffle=False)
results = evaluate_similarity(test_loader, reconstructed_data, input_size, output_size, device)
print(f"Average MSE: {results['mean_mse']}")
print(f"Average SSIM: {results['mean_ssim']}")
```

## Metrics

Reconstruction quality is evaluated using **Mean Squared Error (MSE)** and **Structural Similarity Index (SSIM)**. These metrics compare the reconstructed samples to the class-wise averages of the original test set. Use `amulet.data_reconstruction.metrics.evaluate_similarity` to calculate these scores. Pass it a `DataLoader` built with `batch_size=1`, since it accumulates per-class averages one sample at a time.
