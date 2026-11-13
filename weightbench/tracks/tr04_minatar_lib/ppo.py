"""PPO baseline for TR04-GAME-001..003 (SPEC §4 floor: "PPO (100M–500M steps)"). Server-side, CPU only.

    python -m weightbench.tracks.tr04_minatar_lib.ppo --art ART --steps 25000000 --out ART/private/ppo_info.json

The actor is exactly the task's ``600 → 32 → 32 → 6`` ReLU shell (``spec.PARAM_SHAPES``), so the trained policy is a
valid TR-04 submission and is scored by the same evaluator on the same private eval seeds. A separate
``600 → 64 → 64 → 1`` critic is used for training only. Hyper-parameters follow CleanRL's MinAtar PPO defaults.
Training uses its own seed stream (``spec.task_seeds(seed)["dev"]``), never the eval seeds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import rollout as R
from . import spec as S

N_ENVS, N_STEPS = 16, 128
GAMMA, GAE_LAMBDA = 0.99, 0.95
EPOCHS, MINIBATCHES = 4, 4
CLIP, ENT_COEF, VF_COEF, MAX_GRAD_NORM, LR = 0.2, 0.01, 0.5, 0.5, 2.5e-4


def gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: np.ndarray,
        last_done: np.ndarray, gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation over [T, N] arrays. ``dones[t]`` = episode ended *before* step t's obs."""
    t_max = len(rewards)
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1])
    for t in reversed(range(t_max)):
        next_nonterminal = 1.0 - (last_done if t == t_max - 1 else dones[t + 1])
        next_value = last_value if t == t_max - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last = delta + gamma * lam * next_nonterminal * last
        adv[t] = last
    return adv, adv + values


def _networks(seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    actor = nn.Sequential(nn.Linear(S.N_IN, S.HIDDEN[0]), nn.ReLU(), nn.Linear(S.HIDDEN[0], S.HIDDEN[1]), nn.ReLU(),
                          nn.Linear(S.HIDDEN[1], S.N_ACTIONS))
    critic = nn.Sequential(nn.Linear(S.N_IN, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    return actor, critic


def actor_weights(actor) -> dict[str, np.ndarray]:
    linears = [m for m in actor if hasattr(m, "weight")]
    out = {}
    for i, m in enumerate(linears, 1):
        out[f"W{i}"] = m.weight.detach().numpy().astype(np.float32).copy()
        out[f"b{i}"] = m.bias.detach().numpy().astype(np.float32).copy()
    return out


def train(game: str, total_steps: int, seed: int, log_every: int = 50) -> tuple[dict[str, np.ndarray], list[dict]]:
    import torch

    torch.set_num_threads(4)
    actor, critic = _networks(seed)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=LR, eps=1e-5)
    envs = [R.make_env(game, seed + i) for i in range(N_ENVS)]
    obs = np.stack([R.encode(game, e) for e in envs])
    done = np.zeros(N_ENVS)
    ep_ret, ep_returns, log = np.zeros(N_ENVS), [], []
    n_updates = total_steps // (N_ENVS * N_STEPS)
    for update in range(n_updates):
        for g in opt.param_groups:
            g["lr"] = LR * (1.0 - update / n_updates)
        b_obs = np.zeros((N_STEPS, N_ENVS, S.N_IN), np.float32)
        b_act, b_logp, b_rew, b_val, b_done = (np.zeros((N_STEPS, N_ENVS)) for _ in range(5))
        for t in range(N_STEPS):
            b_obs[t], b_done[t] = obs, done
            with torch.no_grad():
                x = torch.from_numpy(obs)
                dist = torch.distributions.Categorical(logits=actor(x))
                action = dist.sample()
                b_logp[t] = dist.log_prob(action).numpy()
                b_val[t] = critic(x).squeeze(-1).numpy()
            b_act[t] = action.numpy()
            for i, env in enumerate(envs):
                reward, terminal = env.act(int(b_act[t, i]))
                b_rew[t, i] = reward
                ep_ret[i] += reward
                done[i] = float(terminal)
                if terminal:
                    ep_returns.append(ep_ret[i])
                    ep_ret[i] = 0.0
                    env.reset()
                obs[i] = R.encode(game, env)
        with torch.no_grad():
            last_value = critic(torch.from_numpy(obs)).squeeze(-1).numpy()
        adv, ret = gae(b_rew, b_val, b_done, last_value, done)
        f_obs = torch.from_numpy(b_obs.reshape(-1, S.N_IN))
        f_act = torch.from_numpy(b_act.reshape(-1)).long()
        f_logp, f_adv, f_ret = (torch.from_numpy(a.reshape(-1)).float() for a in (b_logp, adv, ret))
        batch = len(f_obs)
        mb = batch // MINIBATCHES
        for _ in range(EPOCHS):
            perm = torch.randperm(batch)
            for start in range(0, batch, mb):
                idx = perm[start:start + mb]
                dist = torch.distributions.Categorical(logits=actor(f_obs[idx]))
                ratio = torch.exp(dist.log_prob(f_act[idx]) - f_logp[idx])
                a = f_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg_loss = torch.max(-a * ratio, -a * torch.clamp(ratio, 1 - CLIP, 1 + CLIP)).mean()
                v_loss = 0.5 * ((critic(f_obs[idx]).squeeze(-1) - f_ret[idx]) ** 2).mean()
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                opt.step()
        if update % log_every == 0 or update == n_updates - 1:
            recent = ep_returns[-100:]
            log.append({"steps": (update + 1) * N_ENVS * N_STEPS,
                        "train_return_100": float(np.mean(recent)) if recent else None})
    return actor_weights(actor), log


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--steps", type=int, default=25_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    art = Path(a.art)
    sp = json.loads((art / "private" / "spec.json").read_text())
    t0 = time.time()
    weights, log = train(sp["game"], a.steps, sp["seeds"]["dev"])
    np.savez(art / "private" / "ppo.npz", **weights)
    returns = R.episode_returns(sp["game"], R.policy_act_fn(sp["game"], weights), sp["seeds"]["eval"],
                                sp["eval_episodes"], sp["max_steps"])
    out = {"game": sp["game"], "steps": a.steps, "eval": R.summarize(returns), "train_log": log,
           "seconds": round(time.time() - t0, 1)}
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
