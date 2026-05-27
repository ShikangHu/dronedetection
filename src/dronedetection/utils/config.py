"""Config loading helpers (Hydra / OmegaConf)."""
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    return OmegaConf.load(path)


def save_config(cfg: DictConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def merge_configs(*cfgs: DictConfig) -> DictConfig:
    return OmegaConf.merge(*cfgs)
