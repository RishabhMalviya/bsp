"""Smoke test for eval video rendering.

The pretraining trainer's `_eval` builds an RGB-array env, calls `env.render()`
once per step, stacks the frames, and transposes them to the (T, C, H, W) layout
that `wandb.Video` expects. This test exercises that exact pipeline with a random
policy (no agent/checkpoint needed) so a regression in env construction, the
render mode, the frame shape, or the transpose surfaces here.

Runnable directly (`python tests/test_eval_video_smoke.py`) or via pytest.
"""

import numpy as np
import wandb

from bsp.common.utils import make_env


# render_kwargs in make_env request height=480, width=640; dm-control returns
# frames as (height, width, 3).
EXPECTED_FRAME_SHAPE = (480, 640, 3)


def _render_eval_video(num_steps=5):
    """Mirror the trainer's _eval frame-collection + stacking, random policy."""
    env = make_env('humanoid', 'run', max_timesteps=num_steps, seed=0, render_mode='rgb_array')
    try:
        env.reset(seed=0)
        frames = [env.render()]
        for _ in range(num_steps):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
        video = np.stack(frames).transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        return frames, video
    finally:
        env.close()


def test_render_returns_valid_rgb_frames():
    frames, _ = _render_eval_video()
    for frame in frames:
        assert isinstance(frame, np.ndarray), type(frame)
        assert frame.shape == EXPECTED_FRAME_SHAPE, frame.shape
        assert frame.dtype == np.uint8, frame.dtype


def test_stacked_video_has_channel_first_layout():
    frames, video = _render_eval_video()
    T, C, H, W = video.shape
    assert T == len(frames), (T, len(frames))
    assert (C, H, W) == (3, 480, 640), video.shape
    assert video.dtype == np.uint8, video.dtype


def test_wandb_video_accepts_rendered_array():
    """The stacked array must be consumable by wandb.Video, the final sink."""
    _, video = _render_eval_video()
    wandb.Video(video, fps=30, format='mp4')


def _run_all():
    tests = [
        test_render_returns_valid_rgb_frames,
        test_stacked_video_has_channel_first_layout,
        test_wandb_video_accepts_rendered_array,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}", flush=True)
    print("ALL EVAL VIDEO SMOKE TESTS PASSED", flush=True)


if __name__ == '__main__':
    _run_all()
