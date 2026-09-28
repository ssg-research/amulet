"""`setup_assets.py` leaves the experiments nothing to fetch or preprocess.

A smoke or full run should spend its time training. Anything it downloads or
preprocesses on first use is time the documented runtimes do not include, and a
network dependency in the middle of a sweep.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import setup_assets
from PIL import Image

from experiments.e1_attack_baselines.context import (
    DEFAULT_TARGET_ATTRIBUTE,
    PRIVACY_TARGET_ATTRIBUTE,
)
from experiments.e5_textbadnets.llm_backdoor_common import SMOKE_MODEL_NAME
from scarab.utils import load_data

# The paper's E5 target, the default `--model_name` of onion.py and dp.py.
FULL_MODEL_NAME = "meta-llama/Llama-3.2-3B"

_CELEBA_ATTRIBUTES = [DEFAULT_TARGET_ATTRIBUTE, PRIVACY_TARGET_ATTRIBUTE, "Male"]


def _plant_raw_celeba(celeba_dir: Path) -> None:
    """Write a tiny CelebA in the raw on-disk layout the loader reads.

    Eight 218x178 JPEGs and an attribute file in CelebA's own format: a count
    line, a header of attribute names, then one `filename value...` row per
    image with values in {-1, 1}. Every attribute is balanced so a stratified
    split works for any target.
    """
    images = celeba_dir / "img_align_celeba"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(8):
        name = f"{i + 1:06d}.jpg"
        pixels = rng.integers(0, 256, (218, 178, 3), dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(images / name)
        values = [1 if (i >> bit) & 1 else -1 for bit in range(len(_CELEBA_ATTRIBUTES))]
        rows.append(" ".join([name, *(str(v) for v in values)]))
    text = "\n".join(["8", " ".join(_CELEBA_ATTRIBUTES), *rows]) + "\n"
    _ = (celeba_dir / "list_attr_celeba.txt").write_text(text)


def test_celeba_asset_leaves_no_e1_target_to_preprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _plant_raw_celeba(tmp_path / "data" / "celeba")
    monkeypatch.setattr(setup_assets, "repo_root", lambda: tmp_path)
    _ = setup_assets.ASSETS_BY_NAME["celeba"].fetch()
    _ = capsys.readouterr()

    for target in (DEFAULT_TARGET_ATTRIBUTE, PRIVACY_TARGET_ATTRIBUTE):
        _ = load_data(tmp_path, "celeba", celeba_target=target)

    assert "Processing CelebA" not in capsys.readouterr().out


def test_llm_assets_download_every_e5_model_with_its_weights(mocker) -> None:
    _ = mocker.patch("datasets.load_dataset", return_value=[])
    snapshot = mocker.patch("huggingface_hub.snapshot_download", return_value="cache")

    for asset in setup_assets.ASSETS:
        if asset.needs_llm:
            _ = asset.fetch()

    with_weights = {
        call.kwargs["repo_id"]
        for call in snapshot.call_args_list
        if call.kwargs.get("allow_patterns") is None
    }
    assert {SMOKE_MODEL_NAME, FULL_MODEL_NAME} <= with_weights
