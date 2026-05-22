"""Training entry point.

Hydra parses the YAML config from `configs/` and applies any CLI overrides
(e.g. `python -m bsp.main seed=1 env.task=balance`). Pick a different file
with `--config-name=<name>`.
"""

import hydra
from omegaconf import DictConfig

from bsp.logger import Logger
from bsp.trainer import Trainer


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = Logger(cfg)
    trainer = Trainer(cfg, logger)
    try:
        for _ in range(cfg.train.num_iterations):
            trainer.train(cfg)
            trainer.eval(cfg)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
