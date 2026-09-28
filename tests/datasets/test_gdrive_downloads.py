"""Fast tests that the Google Drive-backed loaders never show the Drive link.

A Drive file's page shows its owner to anyone who opens it, so the link must not
reach the user: neither gdown's own `From:` line, nor its error message, which
tells the user to open the link in a browser when Drive refuses a download.
"""

import sys
import traceback
from collections.abc import Callable
from pathlib import Path

import pytest
import requests
from gdown.exceptions import FileURLRetrievalError

from scarab.datasets import load_celeba, load_census, load_lfw, load_utkface


def _drive_refusal(url: str) -> Exception:
    return FileURLRetrievalError(f"Open {url} in a browser instead.")


def _network_failure(url: str) -> Exception:
    return requests.exceptions.ConnectionError(f"Max retries exceeded with url: {url}")


@pytest.fixture(params=[_drive_refusal, _network_failure], ids=lambda f: f.__name__)
def failing_drive(request: pytest.FixtureRequest, mocker):
    """Stand in for gdown against a Drive download that fails.

    Like gdown, it prints the link unless `quiet`, then raises an error naming it.
    """
    make_error: Callable[[str], Exception] = request.param

    def _download(id: str, output: str, quiet: bool = False) -> None:
        url = f"https://drive.google.com/uc?id={id}"
        if not quiet:
            print(f"From: {url}", file=sys.stderr)
        raise make_error(url)

    return mocker.patch("gdown.download", side_effect=_download)


@pytest.mark.parametrize(
    "loader",
    [load_celeba, load_census, load_lfw, load_utkface],
    ids=lambda f: f.__name__,
)
def test_failed_download_never_shows_drive_link(
    tmp_path: Path,
    failing_drive,
    capsys: pytest.CaptureFixture[str],
    loader: Callable[[Path], object],
) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        loader(tmp_path)

    captured = capsys.readouterr()
    shown = (
        captured.out + captured.err + "".join(traceback.format_exception(excinfo.value))
    ).lower()
    assert "drive.google.com" not in shown
    assert "google drive" not in shown
