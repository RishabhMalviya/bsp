"""Training entry point.

Hydra parses the YAML config from `configs/` and applies any CLI overrides
(e.g. `python -m bsp.main seed=1 env.task=balance`). Pick a different file
with `--config-name=<name>`.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from bsp.common.utils import Logger, set_seed
from bsp.pretraining.trainer import BodySchemaTrainer


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    logger = Logger(cfg, name_prefix='pretraining')

    try:
        pretrainer = BodySchemaTrainer(cfg, logger)
        pretrainer.train()
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
