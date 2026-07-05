"""Shared fixtures for dataset-loader tests.

The mocked-download tests plant tiny synthetic raw files; every loader needs
real JPEG bytes for that (the loaders decode with PIL), so the encoder helper
lives here.
"""

import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def make_jpeg_bytes():
    """Return a callable producing random RGB JPEG bytes of a given size."""

    def _make(rng: np.random.Generator, height: int, width: int) -> bytes:
        buf = io.BytesIO()
        img = Image.fromarray(
            rng.integers(0, 256, (height, width, 3), dtype=np.uint8), mode="RGB"
        )
        img.save(buf, format="JPEG")
        return buf.getvalue()

    return _make
