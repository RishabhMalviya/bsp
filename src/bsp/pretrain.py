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
from bsp.pretraining.sb3_ppo_trainer import SB3PPOPretrainer


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    logger = Logger(cfg, name_prefix='pretraining')

    try:
        agent_type = cfg.curiosity_pre_training.get('agent_type', 'sb3_ppo')
        if agent_type == 'sb3_ppo':
            pretrainer = SB3PPOPretrainer(cfg, logger)
        else:
            pretrainer = BodySchemaTrainer(cfg, logger)
        pretrainer.train()
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
