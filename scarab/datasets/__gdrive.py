"""Google Drive downloads that never show the file's Drive link."""

from pathlib import Path

import gdown
import requests
from gdown.exceptions import FileURLRetrievalError


def gdrive_download(file_id: str, output: Path, label: str) -> None:
    """Download a Drive file to `output`, printing only its `label`.

    Raises:
        RuntimeError: If Drive refuses the download or the network fails.
    """
    print(f"Downloading {label}...")
    try:
        _ = gdown.download(id=file_id, output=str(output), quiet=True)
    except (FileURLRetrievalError, requests.exceptions.RequestException):
        raise RuntimeError(
            f"Could not download {label}. Check the connection and try again later."
        ) from None
