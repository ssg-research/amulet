# Data Poisoning

Data poisoning attacks involve an adversary injecting malicious samples into a model's training set. Amulet implements the BadNets backdoor attack for image/tabular data and a textual variant (`TextBadNets`) for language models. It provides an outlier-removal defense based on KNN Shapley values, and (for the textual attack) the ONION perplexity-based defense.

Both poisoning defenses follow the same shape and share the `PoisoningDefense` base class. They expose `train_robust()`, which cleans the (poisoned) training set, then retrains the victim on the cleaned data and returns it. `OutlierRemoval` drops low-Shapley outlier samples, and `ONION` removes perplexity-outlier trigger words. This mirrors the library-wide convention that every defense implements its risk's training entry point (`train_robust` / `train_private` / `train_fair`). `ONION` additionally exposes `purify(dataset)` to clean inputs at test time, alongside `train_robust()`.

## Attack

To run a data poisoning attack, use `amulet.poisoning.attacks.BadNets`. This attack embeds a trigger into a portion of the training set and relabels those samples to a target class.

```python
import torch
from torch.utils.data import DataLoader
from amulet.poisoning.attacks import BadNets
from amulet.utils import load_data, initialize_model, train_classifier, get_accuracy

root_dir = './'
dataset_name = 'cifar10'
batch_size = 256
device = 'cuda:0'
epochs = 10
trigger_label = 0
poisoned_portion = 0.1
random_seed = 42

# 1. Load data
data = load_data(root_dir, dataset_name)
target_train_loader = DataLoader(data.train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data.test_set, batch_size=batch_size, shuffle=False)

# 2. Configure Poisoning Attack
poisoning = BadNets(
    trigger_label=trigger_label,
    portion=poisoned_portion,
    random_seed=random_seed,
    dataset_type="image" # or "tabular" for census
)

# 3. Poison the Training Set
poisoned_train_set = poisoning.poison_train(data.train_set)
poisoned_train_loader = DataLoader(poisoned_train_set, batch_size=batch_size, shuffle=True)

# 4. Train Poisoned Model
poisoned_model = initialize_model("vgg", "m1", data.num_features, data.num_classes).to(device)
optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

poisoned_model = train_classifier(
    poisoned_model, poisoned_train_loader, criterion, optimizer, epochs, device
)

# 5. Evaluate Attack Success (on poisoned test set)
poisoned_test_set = poisoning.poison_test(data.test_set)
poisoned_test_loader = DataLoader(poisoned_test_set, batch_size=batch_size, shuffle=False)

clean_acc = get_accuracy(poisoned_model, test_loader, device)
attack_success = get_accuracy(poisoned_model, poisoned_test_loader, device)

print(f"Clean Test Accuracy: {clean_acc}%")
print(f"Attack Success Rate (ASR): {attack_success}%")
```

## Defense

To defend against poisoning, use `amulet.poisoning.defenses.OutlierRemoval`. This module identifies and removes outliers from the training set using KNN Shapley values before retraining the model.

```python
from amulet.poisoning.defenses import OutlierRemoval

# Initialize Outlier Removal Defense
outlier_removal = OutlierRemoval(
    model=poisoned_model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=poisoned_train_loader,
    test_loader=test_loader,
    device=device,
    percent=10 # Remove the bottom 10% of samples by Shapley score
)

# Train the Robust Model
defended_model = outlier_removal.train_robust()

# Evaluate Improved Robustness
defended_asr = get_accuracy(defended_model, poisoned_test_loader, device)
print(f"ASR after defense: {defended_asr}%")
```

## Textual Backdoor (LLM)

`amulet.poisoning.attacks.TextBadNets` is the NLP analog of `BadNets`: it inserts a rare-word or short-phrase trigger into a fraction of training examples (in string space) and relabels them to a target class. The victim is `HFCausalLM`, a LoRA-adapted HuggingFace causal (decoder-only) LM that keeps its generative base: it classifies (trainable head over the frozen base), scores perplexity (its base LM), and generates. This path requires the optional `llm` extra (`uv sync --extra <cuxxx> --extra llm`).

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from amulet.datasets import load_sst2
from amulet.models import HFCausalLM
from amulet.poisoning.attacks import TextBadNets
from amulet.poisoning.defenses import ONION
from amulet.utils import get_accuracy, train_classifier

device = "cuda:0"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 1. Load a text dataset (tokenized with the victim's tokenizer)
data = load_sst2(path="./data/sst2", tokenizer_name=model_name, max_length=128)

# 2. Poison: trigger + relabel a fraction of train; trigger every non-target test row
attack = TextBadNets(trigger="cf", trigger_label=1, portion=0.1, random_seed=0)
poisoned_train = attack.poison_train(data.train_set)
poisoned_test = attack.poison_test(data.test_set)   # accuracy on this vs. trigger_label is ASR

# 3. Fine-tune the LoRA victim on the poisoned data
victim = HFCausalLM(model_name=model_name, num_labels=data.num_classes).to(device)
optimizer = torch.optim.Adam(victim.trainable_parameters(), lr=2e-4)
criterion = torch.nn.CrossEntropyLoss()
train_loader = DataLoader(poisoned_train, batch_size=16, shuffle=True)
victim = train_classifier(victim, train_loader, criterion, optimizer, 3, device)

asr = get_accuracy(victim, DataLoader(poisoned_test, batch_size=16), device)
print(f"Attack Success Rate (ASR): {asr}%")

# 4. Defend with ONION: purify triggered inputs, then re-measure ASR on the same victim.
#    ONION scores perplexity with the victim's own base LM (its clean, pre-fine-tuning
#    base by default), so it re-tokenizes with the victim's tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
onion = ONION(model=victim, tokenizer=tokenizer, device=device)
purified_test = onion.purify(poisoned_test)
defended_asr = get_accuracy(victim, DataLoader(purified_test, batch_size=16), device)
print(f"ASR after ONION: {defended_asr}%")
```

An unintended cross-risk interaction is available by reusing the `DPSGD` membership-inference defense to fine-tune the LoRA victim with per-example clipping and noise (construct the victim with `dtype=torch.float32`, since bf16 trainable parameters break Opacus per-sample clipping). See `examples/attack_pipelines/run_text_backdoor.py` for the full pipeline reporting both the ONION and DP-LoRA interactions.

## Metrics

The primary metric for poisoning attacks is the **Attack Success Rate (ASR)**, which is the model's accuracy on the poisoned test set (i.e., how often it predicts the target trigger label when the trigger is present). Standard classification accuracy on clean data is also monitored to ensure the defense does not degrade performance.
