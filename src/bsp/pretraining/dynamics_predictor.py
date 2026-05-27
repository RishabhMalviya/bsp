import torch
from torch import optim
import torch.nn.functional as F
from omegaconf import DictConfig


from bsp.common.replay_buffer import ReplayBuffer
from bsp.common.utils import get_device
from bsp.pretraining.nn_modules import DynamicsPredictorModule


device = get_device()


class DynamicsPredictor():
    def __init__(self, cfg: DictConfig, obs_dim, ac_dim, H_max):
        self.cfg = cfg
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(obs_dim, ac_dim, cfg.replay_buffer.capacity)

        self.dynamics_predictor_module = DynamicsPredictorModule(
            ac_dim=ac_dim, obs_dim=obs_dim, H_max=H_max,
            **cfg.model,
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
        print(f'Inside compute_intrinsic_reward: obs shape {obs.shape}, ac shape {ac.shape}, next_obs shape {next_obs.shape}')
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
        self.optimizer.step()

        return {
            'dp_state_loss': state_loss.item(),
            'dp_action_loss': action_loss.item(),
            'dp_total_loss': loss.item(),
        }
