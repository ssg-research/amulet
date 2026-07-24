"""Keep the reviewer-facing docs from rotting as the tree changes.

`ARTIFACT.md` and `CLAIMS.md` are the two documents a benchmark reviewer opens
first, and every path they cite is a promise: "this file exists, run this
command against it". A rename elsewhere in the tree can quietly break that
promise, and a reviewer discovering a dead path reads it as a broken artifact.

This test parses both docs and asserts that every repository path they reference
resolves on disk, and that every experiment id they name is registered. It does
not execute any command; it checks that the things the commands point at are
real. Two reference styles are covered:

* **Backtick code spans** hold shell commands, and the tokens inside them that
  look like repository paths (first segment is a known top-level entry) are
  resolved against the repo root, the directory a reviewer runs commands from.
* **Markdown link targets** are resolved against the linking doc's own
  directory, the way a Markdown renderer resolves them.

Placeholder tokens (`runs/<level>/...`, brace or glob forms) are skipped: they
are illustrative, not literal paths.
"""

from __future__ import annotations

import re

import pytest

from common.paths import artifact_root, repo_root
from common.registry import EXPERIMENT_IDS

# Docs under test, relative to the artifact root.
DOC_NAMES: tuple[str, ...] = ("ARTIFACT.md", "CLAIMS.md", "RUNTIME.md")

# A backtick token is treated as a repository path only when its first segment
# is one of these. This excludes API snippets (`Attack(model).attack()`) and
# dotted module paths (`common.io`), which contain no leading known directory.
KNOWN_FIRST_SEGMENTS: frozenset[str] = frozenset({
    "artifact",
    "examples",
    "tests",
    "amulet",
    "docs",
    "pyproject.toml",
    "README.md",
    "AGENTS.md",
    "uv.lock",
})

# Tokens containing any of these are placeholders or shell patterns, not paths.
_PLACEHOLDER_CHARS = "<>*{}"

# Gitignored, run-produced output trees. A doc legitimately names these as where
# output lands, but they do not exist in a fresh checkout, so the integrity
# check skips them rather than demanding a reviewer run something first.
_EPHEMERAL_PREFIXES: tuple[str, ...] = (
    "artifact/runs",
    "artifact/tables/generated",
    "artifact/plots/generated",
)

_LINK = re.compile(r"\]\(([^)]+)\)")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_FENCED_BLOCK = re.compile(r"(?ms)^```.*?\n(.*?)^```")
# A lowercase experiment id such as `e5_textbadnets`, wherever it appears.
_EXPERIMENT_ID = re.compile(r"\be[1-5]_[a-z_]+\b")


def _clean_token(token: str) -> str:
    """Strip surrounding shell/prose punctuation from a candidate path token."""
    return token.strip().strip("\"'").rstrip(").,:;").lstrip("(")


def _tokens_to_paths(tokens: list[str], candidates: list[str]) -> None:
    """Append whitespace tokens that look like repository paths to `candidates`."""
    for raw in tokens:
        token = _clean_token(raw)
        if not token or any(char in token for char in _PLACEHOLDER_CHARS):
            continue
        normalized = token.rstrip("/")
        if any(
            normalized == prefix or normalized.startswith(prefix + "/")
            for prefix in _EPHEMERAL_PREFIXES
        ):
            continue
        if token.split("/", 1)[0] in KNOWN_FIRST_SEGMENTS:
            candidates.append(token)


def _split_fenced_blocks(text: str) -> tuple[list[str], str]:
    """Return fenced-code-block bodies and the text with those blocks removed.

    Fenced blocks and inline code spans are parsed separately: a naive backtick
    scan pairs a fence's backticks with later inline ones and misaligns every
    span after the first block. Splitting them keeps both readings correct.
    """
    bodies = _FENCED_BLOCK.findall(text)
    return bodies, _FENCED_BLOCK.sub("", text)


def _path_candidates(text: str) -> list[str]:
    """Return repository-path tokens from fenced blocks and inline code spans."""
    candidates: list[str] = []
    fenced, remainder = _split_fenced_blocks(text)
    for body in fenced:
        _tokens_to_paths(body.split(), candidates)
    for span in _INLINE_CODE.findall(remainder):
        _tokens_to_paths(span.split(), candidates)
    return candidates


def _link_targets(text: str) -> list[str]:
    """Return local (non-URL, non-anchor) Markdown link targets."""
    targets: list[str] = []
    for raw in _LINK.findall(text):
        target = raw.split("#", 1)[0].strip()
        if not target or target.startswith("#"):
            continue
        if "://" in target or target.startswith("mailto:"):
            continue
        if any(char in target for char in _PLACEHOLDER_CHARS):
            continue
        targets.append(target)
    return targets


def _referenced_paths() -> list[tuple[str, str]]:
    """Collect `(doc_name, referenced_path)` pairs across both docs.

    Backtick tokens resolve against the repo root; link targets resolve against
    the doc's own directory. Both are returned as repo-root-relative strings for
    a readable parametrize id. A doc that does not exist yet contributes nothing,
    so a separate guard test asserts the docs are present and non-trivial.
    """
    root = repo_root()
    pairs: list[tuple[str, str]] = []
    for name in DOC_NAMES:
        doc = artifact_root() / name
        if not doc.is_file():
            continue
        text = doc.read_text()
        for token in _path_candidates(text):
            resolved = (root / token).resolve()
            pairs.append((name, str(resolved.relative_to(root))))
        for target in _link_targets(text):
            resolved = (doc.parent / target).resolve()
            rel = (
                resolved.relative_to(root)
                if resolved.is_relative_to(root)
                else resolved
            )
            pairs.append((name, str(rel)))
    # Deduplicate while keeping order stable for readable parametrize ids.
    return list(dict.fromkeys(pairs))


def _referenced_experiment_ids() -> list[tuple[str, str]]:
    """Collect `(doc_name, experiment_id)` pairs naming a lowercase experiment id."""
    pairs: list[tuple[str, str]] = []
    for name in DOC_NAMES:
        doc = artifact_root() / name
        if not doc.is_file():
            continue
        for match in _EXPERIMENT_ID.findall(doc.read_text()):
            pairs.append((name, match))
    return list(dict.fromkeys(pairs))


_REFERENCED_PATHS = _referenced_paths()
_REFERENCED_EXPERIMENT_IDS = _referenced_experiment_ids()


def test_the_reviewer_docs_exist_and_reference_repository_paths() -> None:
    """Guard the discovery: absent or path-free docs would make the check vacuous.

    Mirrors the conformance suite's non-vacuity guard: without this, deleting
    `ARTIFACT.md` would turn the parametrized existence check green by collecting
    zero cases.
    """
    for name in DOC_NAMES:
        assert (artifact_root() / name).is_file(), f"missing reviewer doc: {name}"
    assert len(_REFERENCED_PATHS) >= 25, (
        "the reviewer docs reference suspiciously few repository paths "
        f"({len(_REFERENCED_PATHS)}); parsing likely regressed"
    )


@pytest.mark.parametrize(
    ("doc_name", "path"),
    _REFERENCED_PATHS,
    ids=[f"{doc}:{path}" for doc, path in _REFERENCED_PATHS],
)
def test_every_repository_path_referenced_in_the_docs_exists(
    doc_name: str, path: str
) -> None:
    """Every path a reviewer doc names resolves to a real file or directory."""
    assert (repo_root() / path).exists(), (
        f"{doc_name} references {path!r}, which does not exist. Update the doc or "
        f"restore the path so the reviewer instructions stay runnable."
    )


@pytest.mark.parametrize(
    ("doc_name", "experiment_id"),
    _REFERENCED_EXPERIMENT_IDS,
    ids=[f"{doc}:{eid}" for doc, eid in _REFERENCED_EXPERIMENT_IDS],
)
def test_every_experiment_id_named_in_the_docs_is_registered(
    doc_name: str, experiment_id: str
) -> None:
    """Every lowercase experiment id a doc names is a real registry entry."""
    assert experiment_id in EXPERIMENT_IDS, (
        f"{doc_name} names experiment id {experiment_id!r}, absent from the "
        f"registry {EXPERIMENT_IDS}. Fix the typo or register the experiment."
    )
