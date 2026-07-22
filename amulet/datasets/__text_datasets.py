"""Text-modality dataset loaders (SST-2, AG News, IMDB).

These are the first text datasets in Amulet. They diverge from the GDrive 3-step
fallback used by the image loaders in `__image_datasets.py`: text corpora and their
canonical splits are managed by Hugging Face `datasets`, so these load from the HF
hub (with a local cache). That divergence is intentional.

The Hugging Face stack ships in the optional `amuletml[llm]` extra, so its imports
are guarded: `import amulet` still works without the extra, and calling a loader
without it raises a clear "install amuletml[llm]" error.

Each loader tokenizes the raw strings with the target tokenizer, pads `input_ids` to
a fixed per-dataset max sequence length, and retains the raw strings on the returned
`TextTensorDataset` so an input-purification defense (ONION) can re-score and
re-tokenize them. SST-2 uses `validation` as its labelled test split (the public
`test` split has masked `-1` labels).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .__data import AmuletDataset, TextTensorDataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# The Hugging Face stack is the optional `llm` extra. Imports are lazy (inside the
# functions that use them) so `import amulet` works without the extra; this shared
# message is raised when a loader is actually called without the extra installed.
_LLM_INSTALL_HINT = (
    "Text datasets require the optional LLM stack. Install it with "
    "`pip install amuletml[llm]` (or `uv sync --extra llm`)."
)

# The plan's license-free fallback target; its tokenizer produces the input_ids.
_DEFAULT_TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _load_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    """Load a Hugging Face tokenizer and ensure it has a pad token.

    Decoder tokenizers (Llama family) often ship without a pad token; padding to a
    fixed length needs one, so fall back to the eos token as Hugging Face recommends.

    Args:
        tokenizer_name: Hub id of the tokenizer to load.

    Returns:
        The loaded tokenizer with `pad_token` set.

    Raises:
        ImportError: If the optional `llm` extra is not installed.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(_LLM_INSTALL_HINT) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # pyright: ignore[reportUnknownMemberType]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenize(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> torch.Tensor:
    """Tokenize strings to a padded `(N, max_length)` LongTensor of `input_ids`."""
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded.input_ids


def _split_slice(split: str, max_samples: int | None) -> str:
    """Build a Hugging Face split string, optionally truncated to `max_samples` rows."""
    return split if max_samples is None else f"{split}[:{max_samples}]"


def _load_hf_classification(
    hub_id: str,
    text_field: str,
    train_split: str,
    test_split: str,
    num_classes: int,
    path: str | Path,
    tokenizer_name: str,
    max_length: int,
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> AmuletDataset:
    """Load a Hugging Face text-classification corpus as a text `AmuletDataset`.

    Args:
        hub_id: Namespaced Hugging Face hub repo id (e.g. "stanfordnlp/sst2").
        text_field: Name of the raw-text column in the corpus.
        train_split: Split name used for training data.
        test_split: Split name used for (labelled) test data.
        num_classes: Number of output classes.
        path: Directory where the dataset is stored or downloaded (the HF cache dir).
        tokenizer_name: Hub id of the tokenizer that produces `input_ids`.
        max_length: Fixed sequence length; `input_ids` are padded/truncated to it.
        max_train_samples: Optional cap on training rows, for tractable runs and tests.
        max_test_samples: Optional cap on test rows.

    Returns:
        An `AmuletDataset` with `modality="text"` whose `train_set`/`test_set`
        are `TextTensorDataset` instances.

    Raises:
        ImportError: If the optional `llm` extra is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(_LLM_INSTALL_HINT) from exc

    resolved_cache = str(Path(path))
    tokenizer = _load_tokenizer(tokenizer_name)

    def _build(split: str, max_samples: int | None) -> TextTensorDataset:
        raw = load_dataset(  # pyright: ignore[reportUnknownMemberType]
            hub_id, split=_split_slice(split, max_samples), cache_dir=resolved_cache
        )
        texts = [str(t) for t in raw[text_field]]
        labels = torch.as_tensor(list(raw["label"]), dtype=torch.long)
        input_ids = _tokenize(texts, tokenizer, max_length)
        return TextTensorDataset(input_ids, labels, texts, tokenizer_name)

    train_set = _build(train_split, max_train_samples)
    test_set = _build(test_split, max_test_samples)

    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=max_length,
        num_classes=num_classes,
        modality="text",
    )


def load_sst2(
    path: str | Path = Path("./data/sst2"),
    tokenizer_name: str = _DEFAULT_TOKENIZER,
    max_length: int = 128,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> AmuletDataset:
    """Load SST-2 (binary sentiment) as a text `AmuletDataset`.

    Uses the namespaced `stanfordnlp/sst2` repo (`datasets>=4` dropped the legacy
    `glue` script loader). The `sentence` field holds the text and `label` is
    0/1. The labelled `validation` split is used as the test set because the public
    `test` split has masked `-1` labels.

    Args:
        path: Directory where the dataset is stored or downloaded.
        tokenizer_name: Hub id of the tokenizer that produces `input_ids`.
        max_length: Fixed sequence length for `input_ids`.
        max_train_samples: Optional cap on training rows.
        max_test_samples: Optional cap on test rows.

    Returns:
        An `AmuletDataset` with `modality="text"`.
    """
    return _load_hf_classification(
        hub_id="stanfordnlp/sst2",
        text_field="sentence",
        train_split="train",
        test_split="validation",
        num_classes=2,
        path=path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )


def load_agnews(
    path: str | Path = Path("./data/agnews"),
    tokenizer_name: str = _DEFAULT_TOKENIZER,
    max_length: int = 256,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> AmuletDataset:
    """Load AG News (4-class topic classification) as a text `AmuletDataset`.

    Uses the namespaced `fancyzhx/ag_news` repo. The `text` field holds the text
    and `label` is 0-3, over the standard `train`/`test` splits.

    Args:
        path: Directory where the dataset is stored or downloaded.
        tokenizer_name: Hub id of the tokenizer that produces `input_ids`.
        max_length: Fixed sequence length for `input_ids`.
        max_train_samples: Optional cap on training rows.
        max_test_samples: Optional cap on test rows.

    Returns:
        An `AmuletDataset` with `modality="text"`.
    """
    return _load_hf_classification(
        hub_id="fancyzhx/ag_news",
        text_field="text",
        train_split="train",
        test_split="test",
        num_classes=4,
        path=path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )


def load_imdb(
    path: str | Path = Path("./data/imdb"),
    tokenizer_name: str = _DEFAULT_TOKENIZER,
    max_length: int = 512,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> AmuletDataset:
    """Load IMDB (binary sentiment, long reviews) as a text `AmuletDataset`.

    Uses the namespaced `stanfordnlp/imdb` repo. The `text` field holds the review
    and `label` is 0/1, over the standard `train`/`test` splits. Its longer
    documents make a buried trigger a harder perplexity outlier for ONION, so IMDB is
    a genuine attack/defense-difficulty knob rather than a redundant sentiment set.

    Args:
        path: Directory where the dataset is stored or downloaded.
        tokenizer_name: Hub id of the tokenizer that produces `input_ids`.
        max_length: Fixed sequence length for `input_ids`.
        max_train_samples: Optional cap on training rows.
        max_test_samples: Optional cap on test rows.

    Returns:
        An `AmuletDataset` with `modality="text"`.
    """
    return _load_hf_classification(
        hub_id="stanfordnlp/imdb",
        text_field="text",
        train_split="train",
        test_split="test",
        num_classes=2,
        path=path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )
