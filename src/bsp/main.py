"""Training entry point.

Hydra parses the YAML config from `configs/` and applies any CLI overrides
(e.g. `python -m bsp.main seed=1 env.task=balance`). Pick a different file
with `--config-name=<name>`.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict

from bsp.common.utils import Logger, set_seed
from bsp.pretraining.trainer import BodySchemaTrainer
from bsp.finetuning.trainer import TaskSpecificTrainer


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    logger = Logger(cfg)

    try:
        # Pretraining: learn the DynamicsTransformer via curiosity-driven exploration.
        pretrainer = BodySchemaTrainer(cfg, logger)
        pretrainer.train()

        # Finetuning: warm-start a task-specific policy from the pretrained
        # DynamicsTransformer checkpoint and train it on the downstream task.
        ckpt_path = Path(cfg.log_dir) / 'checkpoints' / logger.run.id / 'dynamics_transformer.pth'
        with open_dict(cfg):
            cfg.task_training.dpt_checkpoint_path = str(ckpt_path)

        finetuner = TaskSpecificTrainer(cfg, logger)
        finetuner.train()
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
