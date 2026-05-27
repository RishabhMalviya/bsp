SESSION ?= bsp
LOG ?= runs/bsp.log

.PHONY: smoke run

smoke:
	HYDRA_FULL_ERROR=1 uv run python -m bsp.main \
	  curiosity_pre_training.total_num_episodes=4 \
	  curiosity_pre_training.num_collections_per_loop=1 \
	  curiosity_pre_training.dynamics_predictor_utd=1 \
	  curiosity_pre_training.curiosity_agent_utd=1 \
	  curiosity_pre_training.dp_transformer.training.batch_size=32 \
	  curiosity_pre_training.curiosity_agent.batch_size=32 \
	  wandb.mode=disabled

run:
	mkdir -p $(dir $(LOG))
	tmux new-session -d -s $(SESSION) "HYDRA_FULL_ERROR=1 uv run python -m bsp.main 2>&1 | tee $(LOG); exec bash"
	@echo "Started in tmux session '$(SESSION)'."
	@echo "  Attach:   tmux attach -t $(SESSION)"
	@echo "  Tail log: tail -f $(LOG)"
	@echo "  Kill:     tmux kill-session -t $(SESSION)"
