"""Repo paths and config loading, so every script finds files the same way."""

from pathlib import Path

import yaml

# src/utils/paths.py -> parents[2] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    """Read config/config.yaml and return it as a dict."""
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def get_dir(key: str) -> Path:
    """Return a folder from config `paths` (e.g. 'raw'), creating it if missing."""
    path = REPO_ROOT / load_config()["paths"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path
