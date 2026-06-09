# bsp

This project attempts to develop a method to create pretrained RL models through self-supervised training on the robot body, followed by fine-tuning on the specific tasks with that robot body. The hypotehsis is that the pretraining will greatly increase the sample efficiency of training on downstream task-sepcific training (in comparison to training from scratch). 

The environments used to test this hypothesis are taken from the DM Control Suite (via Shimmy + Gymnasium).

## Setup

All you need to do to setup the repository is run this:

```bash
uv pip install -e ".[dev]"
```

Then you should link this to your wandb account for logging of experiments using:

```bash
uv run wandb login
```

## Run

To run experiments, you can use the commands in the Makefile:

### Tests

```bash
make test
```
Runs the smoke tests: finetuning, config, and (with `MUJOCO_GL=egl`) eval-video generation.

### Pretraining

```bash
make pretrain-test   # tiny, fast pretraining run to sanity-check the pipeline
make pretrain        # full pretraining run
```
- `pretrain-test` runs `bsp.pretrain` with minimal settings (4 episodes, batch size 32, UTD 1, frequent video/checkpoint intervals) so you can quickly verify everything works end to end. Note: it still writes to wandb, so afterwards delete the run from wandb to reverse the artifact over-writes.
- `pretrain` launches the full run inside a detached tmux session named `bsp-pretrain`, logging to `runs/bsp-pretrain.log`. The command prints how to attach, tail the log, and kill the session.

### Fine-tuning

```bash
make finetune-test                        # tiny, fast fine-tuning run (wandb disabled)
make finetune-<task>                      # fine-tune on a downstream task, e.g. finetune-stand
make finetune-<task> pretrain_run=<run>   # fine-tune starting from a specific pretrained run
```
- `finetune-test` runs `bsp.finetune` with minimal settings and `wandb.mode=disabled` to check the fine-tuning pipeline.
- `finetune-<task>` runs `bsp.finetune downstream_task=<task>` (the `<task>` matches the part after `finetune-`, e.g. `stand`, `walk`, `run`) inside a detached tmux session named `bsp-<task>`, logging to `runs/bsp-<task>.log`.
- The optional `pretrain_run=<run>` variable selects which pretrained run to start fine-tuning from; it can be passed to both `finetune-test` and `finetune-<task>`.

### Cleanup

```bash
make clean-<task>    # kill the tmux session bsp-<task>
make clean-all       # kill the pretrain, stand, walk, and run sessions
```
- `clean-<task>` kills the tmux session `bsp-<task>`.
- `clean-all` runs cleanup for `pretrain`, `stand`, `walk`, and `run` (errors from missing sessions are ignored).

### Environment Configuration

To change the environment the training interacts with, look into the `configs/env/` folder.
