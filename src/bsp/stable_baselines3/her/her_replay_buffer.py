"""Minimal HerReplayBuffer stub.

The upstream ``off_policy_algorithm`` references ``HerReplayBuffer`` purely in
``isinstance`` / ``issubclass`` guards. Hindsight Experience Replay itself was not
vendored into this project (only ``common`` + ``sac`` are used), so this stub only
needs to be a distinct ``DictReplayBuffer`` subclass for those guards to behave.
Instantiating it raises, to make accidental HER usage loud rather than silently
wrong.
"""

from bsp.stable_baselines3.common.buffers import DictReplayBuffer


class HerReplayBuffer(DictReplayBuffer):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "HerReplayBuffer is not vendored in bsp.stable_baselines3; only the SAC "
            "+ common subset was copied."
        )
