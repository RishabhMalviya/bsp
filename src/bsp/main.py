"""Training entry point.

Hydra parses the YAML config from `configs/` and applies any CLI overrides
(e.g. `python -m bsp.main seed=1 env.task=balance`). Pick a different file
with `--config-name=<name>`.
"""

import hydra
from omegaconf import DictConfig

from bsp.utils import Logger, set_seed
from bsp.common.base_classes import Trainer


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    logger = Logger(cfg)
    trainer = Trainer(cfg, logger)

    try:
        trainer.train(cfg)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
