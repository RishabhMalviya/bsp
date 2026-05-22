"""Weights & Biases logging wrapper."""

from omegaconf import DictConfig, OmegaConf

import wandb


class Logger:
    """Thin wrapper around wandb for logging scalar metrics."""

    def __init__(self, cfg: DictConfig):
        self.run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def log(self, metrics: dict, step: int | None = None) -> None:
        wandb.log(metrics, step=step)

    def finish(self) -> None:
        wandb.finish()
