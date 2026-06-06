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
from bsp.finetuning.sb3_trainer import SB3SACTrainer


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)


    # Check for Pretraining Checkpoint File
    pretrain_run = cfg.get('pretrain_run') or '' # Can be overriden from CLI with `pretrain_run=<run_id>`
    ckpt_path = Path(cfg.log_dir) / 'checkpoints' / pretrain_run / 'dynamics_transformer.pth'
    if not ckpt_path.exists():
        print(f"""WARNING: Checkpoint file not found at {ckpt_path}. Training from scratch.""")
        cfg.task_training.dpt_checkpoint_path = None
    else:
        with open_dict(cfg):
            cfg.task_training.dpt_checkpoint_path = str(ckpt_path)


    # Setup Logger
    if cfg.get('downstream_task') is not None:  # Top-level override from CLI with `downstream_task=<task>`
        with open_dict(cfg):
            cfg.env.downstream_task = cfg.downstream_task
    logger = Logger(cfg, name_prefix=cfg.env.downstream_task)


    try:
        agent_type = cfg.task_training.get('agent_type', 'bsp')
        if agent_type == 'sb3_sac':
            finetuner = SB3SACTrainer(cfg, logger)
        else:
            finetuner = TaskSpecificTrainer(cfg, logger)
        finetuner.train()
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
