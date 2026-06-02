"""Training entry point.

Hydra parses the YAML config from `configs/` and applies any CLI overrides
(e.g. `python -m bsp.main seed=1 env.task=balance`). Pick a different file
with `--config-name=<name>`.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict

from bsp.common.utils import Logger, set_seed
from bsp.finetuning.trainer import TaskSpecificTrainer


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    if cfg.get('downstream_task') is not None:  # Can be overriden from CLI with `downstream_task=<task>`
        with open_dict(cfg):
            cfg.env.downstream_task = cfg.downstream_task

    logger = Logger(cfg, name_prefix=cfg.env.downstream_task)

    pretraining_logger_run_id = cfg.get('pretraining_logger_run_id') or logger.run.id  # Can be overriden from CLI with `pretraining_logger_run_id=<run_id>`
    ckpt_path = Path(cfg.log_dir) / 'checkpoints' / pretraining_logger_run_id / 'dynamics_transformer.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"""Checkpoint file not found at {ckpt_path}. \
                                Try running pretraining first, noting down the run ID, and then running this finetuning step with `pretraining-logger-run-id=<run_id>`.
                                """)
    with open_dict(cfg):
        cfg.task_training.dpt_checkpoint_path = str(ckpt_path)

    try:
        finetuner = TaskSpecificTrainer(cfg, logger)
        finetuner.train()
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
