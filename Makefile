SESSION ?= bsp
LOG ?= runs/bsp.log

.PHONY: test smoke small run run-tasks run-%

test:
	uv run python tests/test_finetuning_smoke.py
	uv run python tests/test_config_smoke.py
	MUJOCO_GL=egl uv run python tests/test_eval_video_smoke.py

smoke:
	HYDRA_FULL_ERROR=1 uv run python -m bsp.main \
	  curiosity_pre_training.total_num_episodes=4 \
	  curiosity_pre_training.num_collections_per_loop=1 \
	  curiosity_pre_training.dynamics_predictor_utd=1 \
	  curiosity_pre_training.curiosity_agent_utd=1 \
	  curiosity_pre_training.dp_transformer.training.batch_size=32 \
	  curiosity_pre_training.curiosity_agent.batch_size=32 \
	  wandb.mode=disabled

small:
	HYDRA_FULL_ERROR=1 uv run python -m bsp.main \
	  curiosity_pre_training.total_num_episodes=50 \
	  curiosity_pre_training.eval_every_episodes=25

run:
	mkdir -p $(dir $(LOG))
	tmux new-session -d -s $(SESSION) "HYDRA_FULL_ERROR=1 uv run python -m bsp.main 2>&1 | tee $(LOG); exec bash"
	@echo "Started in tmux session '$(SESSION)'."
	@echo "  Attach:   tmux attach -t $(SESSION)"
	@echo "  Tail log: tail -f $(LOG)"
	@echo "  Kill:     tmux kill-session -t $(SESSION)"

# Launch one run per downstream task, each in its own tmux session named
# bsp-<task> and logging to runs/bsp-<task>.log.
run-tasks: run-stand run-walk run-run

run-%:
	mkdir -p runs
	tmux new-session -d -s bsp-$* "HYDRA_FULL_ERROR=1 uv run python -m bsp.main downstream_task=$* 2>&1 | tee runs/bsp-$*.log; exec bash"
	@echo "Started downstream task '$*' in tmux session 'bsp-$*'."
	@echo "  Attach:   tmux attach -t bsp-$*"
	@echo "  Tail log: tail -f runs/bsp-$*.log"
	@echo "  Kill:     tmux kill-session -t bsp-$*"
