.PHONY: test pretrain-test pretrain-full finetune-test finetune-all finetune-% clean-all clean-%

test:
	uv run python tests/test_finetuning_smoke.py
	uv run python tests/test_config_smoke.py
	MUJOCO_GL=egl uv run python tests/test_eval_video_smoke.py

############################
# PRETRAINING
############################
pretrain-test: # Delete the run from wandb, and the artifact over-writes will be reversed
	HYDRA_FULL_ERROR=1 uv run python -m bsp.pretrain \
	  curiosity_pre_training.total_num_episodes=4 \
	  curiosity_pre_training.eval_interval=2 \
	  curiosity_pre_training.dynamics_predictor_utd=1 \
	  curiosity_pre_training.curiosity_agent_utd=1 \
	  curiosity_pre_training.dp_transformer.training.batch_size=32 \
	  curiosity_pre_training.curiosity_agent.batch_size=32

pretrain-full:
	mkdir -p runs
	tmux new-session -d -s bsp-pretrain "HYDRA_FULL_ERROR=1 uv run python -m bsp.pretrain 2>&1 | tee runs/bsp-pretrain.log; exec bash"
	@echo "Started in tmux session 'bsp-pretrain'."
	@echo "  Attach:   tmux attach -t bsp-pretrain"
	@echo "  Tail log: tail -f runs/bsp-pretrain.log"
	@echo "  Kill:     tmux kill-session -t bsp-pretrain"


############################
# FINE-TUNING
############################
pretrain_run ?=

finetune-test:  # Delete the run from wandb, and the artifact over-writes will be reversed
	HYDRA_FULL_ERROR=1 uv run python -m bsp.finetune $(if $(pretrain_run),pretrain_run=$(pretrain_run),) \
	  task_training.total_num_episodes=4 \
	  task_training.eval_interval=2 \
	  task_training.utd=1 \
	  task_training.batch_size=32

finetune-all: finetune-stand finetune-walk finetune-run

finetune-%:
	mkdir -p runs
	tmux new-session -d -s bsp-$* "HYDRA_FULL_ERROR=1 uv run python -m bsp.finetune downstream_task=$* $(if $(pretrain_run),pretrain_run=$(pretrain_run),) 2>&1 | tee runs/bsp-$*.log; exec bash"
	@echo "Started downstream task '$*' in tmux session 'bsp-$*'."
	@echo "  Attach:   tmux attach -t bsp-$*"
	@echo "  Tail log: tail -f runs/bsp-$*.log"
	@echo "  Kill:     tmux kill-session -t bsp-$*"


############################
# CLEANUP
############################
clean-all:
	-$(MAKE) clean-pretrain
	-$(MAKE) clean-stand
	-$(MAKE) clean-walk
	-$(MAKE) clean-run

clean-%:
	@echo ""
	@echo "Cleaning up finetuning logs and checkpoints for $*..."
	tmux kill-session -t bsp-$*
	@echo "Done."
