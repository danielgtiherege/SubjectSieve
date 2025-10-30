from __future__ import annotations
import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Config
# -----------------------------

@dataclass
class ControllerConfig:
    # Ladders (bitrate first; FPS coarser)
    bitrate_ladder_kbps: Tuple[int, ...] = (
        250, 500, 750, 1000, 1500, 2000, 2500, 3000,
        3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000,
        7500, 8000, 8500, 9000, 9500, 10000
    )
    fps_ladder: Tuple[int, ...] = (30, 35, 40, 45, 50, 55, 60)

    # Control cadence
    tick_seconds: float = 1.0

    # QoE thresholds (0..6)
    qoe_low: float = 3.2
    qoe_high: float = 4.8
    qoe_improve_eps: float = 0.05

    # Dwell / hysteresis
    min_dwell_seconds: float = 0.8
    greedy_tie_random: bool = True

    # State history length
    history_len: int = 8

    # Reward shaping (prefer bitrate changes over FPS changes)
    smooth_penalty_bitrate: float = 0.15   # ↓ smaller penalty -> easier to change bitrate
    smooth_penalty_fps: float     = 0.45   # ↑ larger penalty -> avoid FPS changes
    low_qoe_penalty: float        = 0.30
    high_qoe_bonus: float         = 0.05

    # Extra shaping to prefer bitrate↓ first when QoE is low
    bonus_bitrate_drop_when_low: float = 0.06
    penalty_fps_drop_when_low_no_bdrop: float = 0.10

    # RL training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    gamma: float = 0.98
    lr: float = 2e-4
    batch_size: int = 128
    replay_capacity: int = 50_000
    warmup_steps: int = 800
    train_steps_per_tick: int = 3
    target_tau: float = 0.005
    eps_start: float = 0.12
    eps_end: float = 0.03
    eps_decay_steps: int = 40_000


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clip(v, lo, hi):
    return max(lo, min(hi, v))


# -----------------------------
# Replay Buffer
# -----------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

    def push(self, s, a, r, s2, d):
        i = self.ptr
        self.state[i] = s
        self.action[i] = a
        self.reward[i] = r
        self.next_state[i] = s2
        self.done[i] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int):
        idx = np.random.randint(0, self.size, size=batch)
        return (self.state[idx],
                self.action[idx],
                self.reward[idx],
                self.next_state[idx],
                self.done[idx])


# -----------------------------
# Q-network
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in = m.weight.size(1)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Controller
# -----------------------------

class QoEOnlyPensieveLikeController:

    def __init__(self, cfg: ControllerConfig):
        set_seed(cfg.seed)
        self.cfg = cfg

        # Build profile grid
        self.profiles: List[Tuple[int, int]] = [
            (b, f) for b in cfg.bitrate_ladder_kbps for f in cfg.fps_ladder
        ]
        self.num_actions = len(self.profiles)

        # Current selection (init top rung)
        self.cur_idx = self._nearest_profile_index(cfg.bitrate_ladder_kbps[-1], cfg.fps_ladder[-1])
        self.last_change_ts = -1e9

        # QoE history
        self.q_hist = [0.0] * cfg.history_len
        self.prev_qoe_raw: Optional[float] = None

        # RL
        self.state_dim = self._make_state(0.0).shape[0]
        self.q = MLP(self.state_dim, self.num_actions).to(cfg.device)
        self.q_target = MLP(self.state_dim, self.num_actions).to(cfg.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_capacity, self.state_dim)
        self.huber = nn.SmoothL1Loss(reduction="mean")
        self.total_steps = 0

    # ---- Public API ----

    def set_encoder_state(self, bitrate_kbps: int, fps: int):
        self.cur_idx = self._nearest_profile_index(bitrate_kbps, fps)
        self.last_change_ts = -1e9  # allow immediate action

    def update(self, obs: Dict) -> Tuple[int, int]:
        """
        Call once per tick. obs must include {"QoE": 0..6}.
        Returns (target_bitrate_kbps, target_fps).
        """
        q_raw = clip(float(obs["QoE"]), 0.0, 6.0)
        q_norm = q_raw / 6.0
        self._push_qoe(q_norm)

        # Build state
        s = self._make_state(q_norm)

        # Reward previous transition & train
        r = 0.0
        if self.prev_qoe_raw is not None:
            r = self._compute_reward(prev_q=self.prev_qoe_raw/6.0, cur_q=q_norm)
        if hasattr(self, "_prev_state") and hasattr(self, "_prev_action"):
            self.replay.push(self._prev_state, self._prev_action, r, s, 0.0)
            if self.replay.size >= self.cfg.warmup_steps:
                for _ in range(self.cfg.train_steps_per_tick):
                    self._train_step()

        # Choose action: masked DDQN
        idx_allowed = self._mask_allowed_actions(q_raw=q_raw)
        a = self._select_action(s, idx_allowed)

        # Apply change
        if a != self.cur_idx:
            self.last_change_ts = time.time()
        self.cur_idx = a

        # Save for next transition
        self._prev_state = s
        self._prev_action = a
        self.prev_qoe_raw = q_raw

        b, f = self.profiles[self.cur_idx]

        return int(b), int(f)

    # ---- Internals ----

    def _nearest_profile_index(self, bitrate_kbps: int, fps: int) -> int:
        b_lad = np.array(self.cfg.bitrate_ladder_kbps)
        f_lad = np.array(self.cfg.fps_ladder)
        bi = int(np.argmin(np.abs(b_lad - bitrate_kbps)))
        fi = int(np.argmin(np.abs(f_lad - fps)))
        return bi * len(f_lad) + fi

    def _idx_to_components(self, idx: int) -> Tuple[int, int, int, int]:
        fcount = len(self.cfg.fps_ladder)
        bi = idx // fcount
        fi = idx % fcount
        b = self.cfg.bitrate_ladder_kbps[bi]
        f = self.cfg.fps_ladder[fi]
        return bi, fi, b, f

    def _push_qoe(self, q_norm: float):
        self.q_hist.pop(0)
        self.q_hist.append(q_norm)

    def _make_state(self, q_norm: float) -> np.ndarray:
        qvec = np.array(self.q_hist, dtype=np.float32)
        dq = float(np.clip(self.q_hist[-1] - self.q_hist[-2], -1.0, 1.0)) if len(self.q_hist) >= 2 else 0.0

        bi, fi, _, _ = self._idx_to_components(self.cur_idx)
        b_pos = bi / max(1, len(self.cfg.bitrate_ladder_kbps) - 1)
        f_pos = fi / max(1, len(self.cfg.fps_ladder) - 1)

        dwell_s = time.time() - self.last_change_ts
        dwell_n = float(np.clip(dwell_s / 5.0, 0.0, 1.0))

        feats = np.concatenate([qvec, np.array([dq, b_pos, f_pos, dwell_n], dtype=np.float32)])
        return feats

    def _mask_allowed_actions(self, q_raw: float) -> np.ndarray:
        allowed = np.ones(self.num_actions, dtype=bool)

        # Dwell: short hold after a change
        dwell_s = (time.time() - self.last_change_ts)
        if dwell_s < self.cfg.min_dwell_seconds:
            allowed[:] = False
            allowed[self.cur_idx] = True
            return allowed

        # Prefer bitrate-first on *down* moves when QoE is low:
        q = clip(q_raw, 0.0, 6.0)
        if q <= self.cfg.qoe_low:
            bi_cur, fi_cur, _, _ = self._idx_to_components(self.cur_idx)
            for k in range(self.num_actions):
                bi_k, fi_k, _, _ = self._idx_to_components(k)
                # Disallow pure FPS-down or FPS-heavier downs if a bitrate down is still available
                pure_fps_down = (fi_k < fi_cur) and (bi_k >= bi_cur)
                if pure_fps_down and bi_cur > 0:
                    allowed[k] = False

        return allowed

    def _select_action(self, s: np.ndarray, allowed: np.ndarray) -> int:
        self.total_steps += 1
        eps = self._epsilon()
        if random.random() < eps:
            idxs = np.flatnonzero(allowed)
            return int(random.choice(idxs.tolist()))

        self.q.eval()
        with torch.no_grad():
            qvals = self.q(torch.from_numpy(s).to(self.cfg.device).unsqueeze(0)).cpu().numpy().squeeze(0)
        qvals_masked = np.where(allowed, qvals, -1e9)
        best = qvals_masked.max()
        if self.cfg.greedy_tie_random:
            near = np.isclose(qvals_masked, best, rtol=0.0, atol=1e-6)
            cand = np.flatnonzero(near & allowed)
            if cand.size > 1:
                return int(random.choice(cand.tolist()))
        return int(qvals_masked.argmax())

    def _epsilon(self) -> float:
        frac = min(1.0, self.total_steps / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac

    # ----- Reward (utility - smoothness, bitrate-first) -----

    def _compute_reward(self, prev_q: float, cur_q: float) -> float:
        
        r = (cur_q - prev_q)

        bi0, fi0, _, _ = self._idx_to_components(getattr(self, "_prev_action", self.cur_idx))
        bi1, fi1, _, _ = self._idx_to_components(self.cur_idx)

        b_span = max(1, len(self.cfg.bitrate_ladder_kbps) - 1)
        f_span = max(1, len(self.cfg.fps_ladder) - 1)
        b_norm_change = abs(bi1 - bi0) / b_span
        f_norm_change = abs(fi1 - fi0) / f_span

        r -= self.cfg.smooth_penalty_bitrate * b_norm_change
        r -= self.cfg.smooth_penalty_fps     * f_norm_change

        cur_q_raw = cur_q * 6.0
        if cur_q_raw < self.cfg.qoe_low:
            gap = (self.cfg.qoe_low - cur_q_raw) / 6.0
            r -= self.cfg.low_qoe_penalty * gap
            # Prefer bitrate↓ first when low QoE
            if bi1 < bi0:
                r += self.cfg.bonus_bitrate_drop_when_low
            if (fi1 < fi0) and (bi1 >= bi0):
                r -= self.cfg.penalty_fps_drop_when_low_no_bdrop

        if cur_q_raw >= self.cfg.qoe_high:
            r += self.cfg.high_qoe_bonus

        return float(r)

    # ----- Train -----

    def _train_step(self):
        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size)
        s_t  = torch.from_numpy(s).to(self.cfg.device)
        a_t  = torch.from_numpy(a).to(self.cfg.device)
        r_t  = torch.from_numpy(r).to(self.cfg.device)
        s2_t = torch.from_numpy(s2).to(self.cfg.device)
        d_t  = torch.from_numpy(d).to(self.cfg.device)

        with torch.no_grad():
            q_next_online = self.q(s2_t)
            a_prime = torch.argmax(q_next_online, dim=1)
            q_next_target = self.q_target(s2_t)
            q_target_sel = q_next_target.gather(1, a_prime.unsqueeze(1)).squeeze(1)
            y = r_t + (1.0 - d_t) * self.cfg.gamma * q_target_sel

        q_sa = self.q(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        loss = self.huber(q_sa, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=5.0)
        self.opt.step()

        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.data.mul_(1.0 - self.cfg.target_tau).add_(self.cfg.target_tau * p.data)

    # ----- Persistence / helpers -----

    def confirm_apply(self, bitrate_kbps: int, fps: int):
        self.cur_idx = self._nearest_profile_index(bitrate_kbps, fps)

    def save(self, path: str):
        torch.save({
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "opt": self.opt.state_dict(),
            "steps": self.total_steps,
            "cur_idx": self.cur_idx,
            "q_hist": self.q_hist,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.opt.load_state_dict(ckpt["opt"])
        self.total_steps = ckpt.get("steps", 0)
        self.cur_idx = ckpt.get("cur_idx", self.cur_idx)
        self.q_hist = ckpt.get("q_hist", self.q_hist)
