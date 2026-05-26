.PHONY: smoke

smoke:
	HYDRA_FULL_ERROR=1 uv run python -m bsp.main \
	  curiosity_pre_training.total_num_episodes=4 \
	  curiosity_pre_training.num_collections_per_loop=1 \
	  curiosity_pre_training.dynamics_training_iterations=1 \
	  curiosity_pre_training.curiosity_training_iterations=1 \
	  curiosity_pre_training.dp_transformer.training.batch_size=32 \
	  curiosity_pre_training.curiosity_agent.batch_size=32 \
	  wandb.mode=disabled
