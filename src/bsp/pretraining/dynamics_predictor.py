import torch
from torch import optim
import torch.nn.functional as F
from omegaconf import DictConfig


from bsp.common.replay_buffer import ReplayBuffer
from bsp.common.utils import get_device
from bsp.pretraining.nn_modules import DynamicsPredictorModule


device = get_device()


class DynamicsPredictor():
    def __init__(self, cfg: DictConfig, obs_dim, ac_dim, H_max, model_cfg: DictConfig):
        self.cfg = cfg

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(obs_dim, ac_dim, cfg.replay_buffer.capacity)

        self.dynamics_predictor_module = DynamicsPredictorModule(
            ac_dim=ac_dim, obs_dim=obs_dim, H_max=H_max,
            **model_cfg, # pyright: ignore[reportCallIssue]
        ).to(device)

        self.optimizer = optim.Adam(self.dynamics_predictor_module.parameters(), lr=cfg.training.lr)


    def compute_intrinsic_reward(self, obs: torch.Tensor, ac: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """
        Per-step intrinsic reward via causal next-state prediction.

        For transition (s_l, a_l, s_{l+1}), the reward is the MSE of predicting
        s_{l+1} given the prefix (s_0, a_0, ..., s_l, a_l).

        Implementation: extend the sequence by one step using next_obs[:, -1, :]
        (= s_L), pad actions at the trailing position (masked at every row), and
        build a (B*L, L+1, ·) batch where row l masks positions p > l in both
        streams. Caller must ensure L + 1 <= H_max so positional embeddings fit.
        """
        B, L, obs_dim = obs.shape
        ac_dim = ac.shape[-1]
        Lp = L + 1

        states_full = torch.cat([obs, next_obs[:, -1:, :]], dim=1)
        ac_pad = torch.zeros(B, 1, ac_dim, device=ac.device, dtype=ac.dtype)
        actions_full = torch.cat([ac, ac_pad], dim=1)

        states_exp = states_full.unsqueeze(1).expand(B, L, Lp, obs_dim).reshape(B * L, Lp, obs_dim)
        actions_exp = actions_full.unsqueeze(1).expand(B, L, Lp, ac_dim).reshape(B * L, Lp, ac_dim)

        # Causal mask: at row l, positions p > l are masked (l itself is visible).
        rows = torch.arange(L, device=obs.device)
        positions = torch.arange(Lp, device=obs.device)
        mask = (positions.unsqueeze(0) > rows.unsqueeze(1))
        mask = mask.unsqueeze(0).expand(B, L, Lp).reshape(B * L, Lp)

        with torch.no_grad():
            pred_states, _ = self.dynamics_predictor_module(states_exp, actions_exp, mask, mask)
            pred_states = pred_states.reshape(B, L, Lp, obs_dim)
            idx = torch.arange(L, device=obs.device)
            pred_next = pred_states[:, idx, idx + 1, :]
            targets = states_full[:, 1:, :]
            rewards = ((pred_next - targets) ** 2).mean(dim=-1)

        return rewards

    def compute_intrinsic_reward_step(self, states, actions, next_state) -> float:
        """Online, single-transition intrinsic reward.

        On-policy algorithms (e.g. PPO) need the reward assigned at collection
        time, one transition at a time, rather than recomputed from sampled
        sequences. This is the single-transition analogue of the *last row* of
        :meth:`compute_intrinsic_reward`: given the in-episode prefix states
        ``s_0..s_l`` and actions ``a_0..a_l`` (``T`` steps) plus the resulting
        next state ``s_{l+1}``, return the MSE of predicting ``s_{l+1}`` from the
        causal prefix.

        Masking convention matches :meth:`compute_intrinsic_reward` exactly: the
        sequence is extended by the target state, the trailing action is padded
        (and masked by the module), and only the final position is masked so the
        prefix ``0..l`` stays visible. Caller must ensure ``T + 1 <= H_max``.
        """
        module_device = next(self.dynamics_predictor_module.parameters()).device
        states_t = torch.as_tensor(states, dtype=torch.float32, device=module_device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=module_device)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=module_device)

        T = states_t.shape[0]

        states_full = torch.cat([states_t, next_state_t.unsqueeze(0)], dim=0).unsqueeze(0)  # (1, T+1, obs)
        ac_pad = torch.zeros(1, actions_t.shape[-1], dtype=torch.float32, device=module_device)
        actions_full = torch.cat([actions_t, ac_pad], dim=0).unsqueeze(0)  # (1, T+1, ac)

        # Causal mask: only the final position (the target s_{l+1}) is hidden;
        # positions 0..T-1 stay visible.
        positions = torch.arange(T + 1, device=module_device)
        mask = (positions > (T - 1)).unsqueeze(0)  # (1, T+1)

        with torch.no_grad():
            pred_states, _ = self.dynamics_predictor_module(states_full, actions_full, mask, mask)
            pred_next = pred_states[0, T]
            reward = ((pred_next - next_state_t) ** 2).mean()

        return float(reward.item())

    def _sample_mlm_masks(self, B, L, device):
        n = max(1, L // 4)
        sm = torch.zeros(B, L, dtype=torch.bool, device=device)
        am = torch.zeros(B, L, dtype=torch.bool, device=device)
        sm.scatter_(1, torch.rand(B, L, device=device).argsort(dim=1)[:, :n], True)
        am.scatter_(1, torch.rand(B, L, device=device).argsort(dim=1)[:, :n], True)
        return sm, am

    def update(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        """
            MLM pretraining: mask max(1, L//4) state positions and max(1, L//4)
            action positions per sample, predict the full sequence, and minimize
            reconstruction MSE only at the masked positions.
        """
        obs, ac = batch
        B, L, _ = obs.shape

        sm, am = self._sample_mlm_masks(B, L, obs.device)

        pred_states, pred_actions = self.dynamics_predictor_module(obs, ac, sm, am)

        state_err = ((pred_states - obs) ** 2).mean(dim=-1)
        action_err = ((pred_actions - ac) ** 2).mean(dim=-1)

        state_loss = (state_err * sm.float()).sum() / sm.float().sum().clamp(min=1.0)
        action_loss = (action_err * am.float()).sum() / am.float().sum().clamp(min=1.0)
        loss = state_loss + action_loss

        self.optimizer.zero_grad()
        loss.backward()

        metrics = {
            'dp_state_loss': state_loss.item(),
            'dp_action_loss': action_loss.item(),
            # 'dp_total_loss': loss.item(),
        }
        if self.dynamics_predictor_module.last_state_slice_grad_norm is not None:
            metrics['dp_state_slice_grad_norm'] = self.dynamics_predictor_module.last_state_slice_grad_norm
        if self.dynamics_predictor_module.last_action_slice_grad_norm is not None:
            metrics['dp_action_slice_grad_norm'] = self.dynamics_predictor_module.last_action_slice_grad_norm

        self.optimizer.step()

        return metrics
