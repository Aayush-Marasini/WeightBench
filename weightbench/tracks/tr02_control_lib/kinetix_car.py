"""Kinetix ``l/h2_car_ramp`` (Matthews et al., 2024, kinetix-env 3.0.2) wrapped for TR02-CTRL-001.

Scene (metres, y up; floor top at y=0.4, walls at x<=0 and x>=5):
  car  : chassis = polygon 5 (0.66 x 0.22 box) + two motorised wheels on revolute joints:
         circle 0 (radius 0.22, GREEN = the "agent" shape, front-left wheel) and circle 1.
  goal : BLUE dynamic box (polygon 4) resting on the floor.
  ramp : static plank (polygon 6), 0.7 long, leaning at +0.95 rad (foot x~2.45, top x~1.95).
  lava : RED static strip (polygon 7), x in [1.18, 2.06] (before jitter), on the floor.
Reward: +1 (and the episode ends) when a green shape touches a blue one; -1 (ends) when green
touches red; plus dense shaping 0.1 * (decrease of the green-blue distance). 256 steps, each step
is 2 physics sub-steps of dt = 1/60 s.
Actions (6, continuous, clipped to [-1, 1]): a[0] drives BOTH wheel motors (+1 = full speed
rolling LEFT, i.e. towards negative x; -1 = full speed right). a[1:4] are unbound motor slots and
a[4:6] unbound thrusters: they have no effect in this scene.

Level variants share the same shape slots, so the symbolic observation layout never changes:
  kind 0 "ramp"  : goal on the left behind ramp + lava (the failure scene)
  kind 1 "right" : ramp + lava on the left as usual, goal on the right of the car
  kind 2 "flat"  : goal on the left, ramp + lava removed (inactive)
A variant row is [kind, car_dx, goal_x, ramp_dx] (car and ramp+lava shifted along x).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

SCENE_DOC = """Scene (metres, x to the right, y up; floor surface at y = 0.4; walls at x <= 0 and x >= 5):

- **car**: chassis = polygon slot 5 (0.66 x 0.22 box, starts at x = 2.8, y = 0.73) with two
  motorised wheels on revolute joints: circle 0 (radius 0.22, GREEN = the agent shape, the left
  wheel, x = 2.54) and circle 1 (radius 0.23, right wheel, x = 3.11).
- **goal**: BLUE dynamic box (polygon slot 4, 0.42 x 0.76) standing on the floor.
- **ramp**: static plank (polygon slot 6, 0.14 x 0.70) centred at (2.23, 0.56), rotated +0.95 rad,
  i.e. a steep ramp rising towards -x: foot near x = 2.45 on the floor, top edge near (1.95, 0.8).
- **lava**: RED static strip (polygon slot 7, 0.88 x 0.10) centred at (1.62, 0.40) on the floor
  between the ramp and the goal.

Reward: +1 and episode end when a green shape touches a blue shape; -1 and episode end when a green
shape touches a red shape; plus dense shaping 0.1 x (decrease of the green-to-blue distance) each step.
Episode = 256 steps; each step applies the action for 2 physics sub-steps of dt = 1/60 s.

Actions: 6 continuous values, clipped to [-1, 1]. `action[0]` drives BOTH wheel motors
(+1 = full speed rolling LEFT towards the goal side, -1 = full speed rolling RIGHT); the motor target
speed is proportional to action[0]. `action[1:4]` are unbound motor slots and `action[4:6]` unbound
thruster slots: they have no effect in this scene.
"""

KIND_RAMP, KIND_RIGHT, KIND_FLAT = 0, 1, 2
N_STEPS = 256
OBS_DIM = 679
ACT_DIM = 6

# variant ranges (agent probes must lie inside them)
RANGES = {
    KIND_RAMP: {"car_dx": (-0.25, 0.7), "goal_x": (0.2, 0.55), "ramp_dx": (-0.06, 0.06)},
    KIND_RIGHT: {"car_dx": (-0.6, 0.0), "goal_x": (4.0, 4.6), "ramp_dx": (-0.06, 0.06)},
    KIND_FLAT: {"car_dx": (-0.8, 0.6), "goal_x": (0.3, 1.0), "ramp_dx": (0.0, 0.0)},
}

# ---- symbolic-flat observation layout ---------------------------------------------------------
# polygons (12 slots, walls 1..3 dropped -> 9 rows x 26), circles (4 x 18), joints (6 x 54),
# thrusters (2 x 24), gravity (1).  Polygon slot p>=4 is row p-3; slot 0 (floor) is row 0.
POLY_FEATS = (["pos_x", "pos_y", "vel_x", "vel_y", "inv_mass", "inv_inertia", "density",
               "tanh(ang_vel/10)", "role0_static", "role1_green", "role2_blue", "role3_red",
               "sin(rot)", "cos(rot)", "friction", "restitution", "is_circle(=0)"]
              + [f"vertex{i // 2}_{'xy'[i % 2]}" for i in range(8)] + ["n_vertices<=3"])
CIRC_FEATS = POLY_FEATS[:16] + ["radius", "is_circle(=1)"]
JOINT_FEATS = (["active", "is_fixed", "from_rel_x", "from_rel_y", "to_rel_x", "to_rel_y", "sin(rot)", "cos(rot)",
                "motor_speed", "motor_power", "motor_on", "has_limits", "bind0", "bind1", "bind2", "bind3",
                "min_sin", "min_cos", "max_sin", "max_cos", "diff_min", "diff_max"]
               + [f"onehot_from_shape{i}" for i in range(16)] + [f"onehot_to_shape{i}" for i in range(16)])
THR_FEATS = (["active", "rel_x", "rel_y", "bind0", "bind1", "sin(rot)", "cos(rot)", "power"]
             + [f"onehot_shape{i}" for i in range(16)])
POLY_ROW = {0: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8}
OFF_CIRC = 9 * 26
OFF_JOINT = OFF_CIRC + 4 * 18
OFF_THR = OFF_JOINT + 6 * 54
OFF_GRAV = OFF_THR + 2 * 24
SHAPE_NAMES = {0: "floor", 4: "goal (blue box)", 5: "car chassis", 6: "ramp", 7: "lava (red strip)"}


def pidx(slot: int, feat: str) -> int:
    return POLY_ROW[slot] * 26 + POLY_FEATS.index(feat)


def cidx(c: int, feat: str) -> int:
    return OFF_CIRC + c * 18 + CIRC_FEATS.index(feat)


CX, CY, CVX, CVY = pidx(5, "pos_x"), pidx(5, "pos_y"), pidx(5, "vel_x"), pidx(5, "vel_y")
CSIN, CCOS = pidx(5, "sin(rot)"), pidx(5, "cos(rot)")
GX = pidx(4, "pos_x")
RAMP_X, RAMP_FRIC = pidx(6, "pos_x"), pidx(6, "friction")
LAVA_X = pidx(7, "pos_x")
DEFAULT_TRACE = {"chassis_x": CX, "chassis_y": CY, "chassis_vx": CVX, "chassis_vy": CVY,
                 "chassis_sin": CSIN, "chassis_cos": CCOS, "goal_x": GX}


def obs_layout_markdown() -> str:
    lines = ["# Observation layout (679 floats, Kinetix SYMBOLIC_FLAT, clipped to [-10, 10])", "",
             "Inactive slots are all zeros. Indices are 0-based.", "",
             "## Polygons: rows of 26 features, `index = row*26 + feature`", "",
             "| row | shape | x idx | features |", "|---|---|---|---|"]
    for slot, row in POLY_ROW.items():
        lines.append(f"| {row} | slot {slot}: {SHAPE_NAMES.get(slot, 'unused (zeros)')} | {row * 26} | "
                     f"{row * 26}..{row * 26 + 25} |")
    lines += ["", "Polygon features (offset: name): " + ", ".join(f"{i}: {f}" for i, f in enumerate(POLY_FEATS)),
              "", "Vertices are in the shape's local frame. Rotation is the body angle (counter-clockwise +).",
              "", f"## Circles: 4 rows of 18 features starting at {OFF_CIRC}", "",
              "| row | shape | first idx |", "|---|---|---|",
              f"| 0 | wheel (GREEN, front-left, global shape index 12) | {OFF_CIRC} |",
              f"| 1 | wheel (rear-right, global shape index 13) | {OFF_CIRC + 18} |",
              f"| 2, 3 | unused | {OFF_CIRC + 36} |", "",
              "Circle features: " + ", ".join(f"{i}: {f}" for i, f in enumerate(CIRC_FEATS)), "",
              f"## Joints: 6 rows of 54 features starting at {OFF_JOINT} (rows 0, 1 = the two wheel motors)", "",
              "Joint features: " + ", ".join(f"{i}: {f}" for i, f in enumerate(JOINT_FEATS)), "",
              f"## Thrusters: 2 rows of 24 starting at {OFF_THR} (unused, zeros)", "",
              f"## Gravity: index {OFF_GRAV} = gravity_y / 10 (= -0.981)", "",
              "## Handy indices", ""]
    for k, v in DEFAULT_TRACE.items():
        lines.append(f"- `{k}` = obs[{v}]")
    lines += [f"- `ramp_x` = obs[{RAMP_X}], `ramp_friction` = obs[{RAMP_FRIC}] (1.0 when the ramp exists, 0 if removed)",
              f"- `lava_x` = obs[{LAVA_X}] (0 if removed)"]
    return "\n".join(lines) + "\n"


# ---- environment --------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _base():
    import kinetix
    from kinetix.environment.env import make_kinetix_env
    from kinetix.environment.spaces import ActionType, ObservationType
    from kinetix.util.saving import load_from_json_file

    lvl = os.path.join(os.path.dirname(kinetix.__file__), "levels", "l", "h2_car_ramp.json")
    state, sp, ep = load_from_json_file(lvl)
    env = make_kinetix_env(ActionType.CONTINUOUS, ObservationType.SYMBOLIC_FLAT, None, ep, sp, auto_reset=False)
    return state, sp, ep, env


def sample_variants(rng: np.random.Generator, kind: int, n: int) -> np.ndarray:
    out = np.zeros((n, 4), np.float32)
    out[:, 0] = kind
    for j, k in enumerate(("car_dx", "goal_x", "ramp_dx")):
        lo, hi = RANGES[kind][k]
        out[:, 1 + j] = rng.uniform(lo, hi, n)
    return out


def validate_variant(v) -> np.ndarray:
    if isinstance(v, dict):
        kind = int(v.get("kind", 0))
        if kind not in RANGES:
            raise ValueError("kind must be 0 (ramp), 1 (right) or 2 (flat)")
        row = [kind] + [float(v.get(k, sum(RANGES[kind][k]) / 2)) for k in ("car_dx", "goal_x", "ramp_dx")]
    else:
        row = [float(x) for x in v]
        if len(row) != 4 or int(row[0]) not in RANGES:
            raise ValueError("variant must be [kind, car_dx, goal_x, ramp_dx]")
    kind = int(row[0])
    for j, k in enumerate(("car_dx", "goal_x", "ramp_dx")):
        lo, hi = RANGES[kind][k]
        if not (lo - 1e-9 <= row[1 + j] <= hi + 1e-9):
            raise ValueError(f"{k}={row[1 + j]} outside [{lo}, {hi}] for kind {kind}")
    return np.array(row, np.float32)


def make_states(variants: np.ndarray):
    import jax
    import jax.numpy as jnp

    state, sp, ep, env = _base()

    def one(v):
        kind, car_dx, goal_x, ramp_dx = v[0], v[1], v[2], v[3]
        P, C = state.polygon, state.circle
        ppos = P.position.at[5, 0].add(car_dx).at[4, 0].set(goal_x)
        ppos = ppos.at[6, 0].add(ramp_dx).at[7, 0].add(ramp_dx)
        cpos = C.position.at[0, 0].add(car_dx).at[1, 0].add(car_dx)
        has = kind != KIND_FLAT
        active = P.active.at[6].set(has).at[7].set(has)
        return state.replace(polygon=P.replace(position=ppos, active=active), circle=C.replace(position=cpos))

    return jax.vmap(one)(jnp.asarray(variants, jnp.float32))


# ---- policies (jax) ---------------------------------------------------------------------------
def mlp_shared(p, o, t):
    import jax.numpy as jnp

    h = jnp.maximum(o @ p["fc0.weight"].T + p["fc0.bias"], 0.0)
    h = jnp.maximum(h @ p["fc1.weight"].T + p["fc1.bias"], 0.0)
    return h @ p["fc2.weight"].T + p["fc2.bias"]


def mlp_batched(p, o, t):
    import jax.numpy as jnp

    h = jnp.maximum(jnp.einsum("bij,bj->bi", p["fc0.weight"], o) + p["fc0.bias"], 0.0)
    h = jnp.maximum(jnp.einsum("bij,bj->bi", p["fc1.weight"], h) + p["fc1.bias"], 0.0)
    return jnp.einsum("bij,bj->bi", p["fc2.weight"], h) + p["fc2.bias"]


def expert_a0(o):
    """Cautious expert: roll towards the goal at half throttle (a[0] > 0 = left)."""
    import jax.numpy as jnp

    return 0.5 * jnp.tanh(4.0 * (o[..., CX] - o[..., GX]))


def noisy_expert(p, o, t):
    """p = {"noise": [B, T]} additive throttle noise (data collection)."""
    import jax.numpy as jnp

    a0 = expert_a0(o) + p["noise"][:, t]
    return jnp.zeros((o.shape[0], ACT_DIM)).at[:, 0].set(a0)


def rollout(policy_fn, params, variants: np.ndarray, record: bool = False) -> dict:
    """policy_fn(params, obs[B, 679], t) -> action[B, 6] (clipped to [-1, 1])."""
    import jax

    states = make_states(variants)
    out = _rollout_jit(policy_fn, record)(params, states)
    return jax.tree.map(np.asarray, out)


@lru_cache(maxsize=8)
def _rollout_jit(policy_fn, record):
    import jax
    import jax.numpy as jnp

    _, sp, ep, env = _base()

    def run(params, states):
        obs0 = jax.vmap(env.get_obs)(states)
        B = obs0.shape[0]

        def step(carry, t):
            s, obs, alive, ret, succ, lava, tend = carry
            a = jnp.clip(policy_fn(params, obs, t), -1.0, 1.0)
            o2, s2, r, d, _ = jax.vmap(lambda st, ac: env.step(jax.random.PRNGKey(0), st, ac, ep))(s, a)
            r = jnp.where(alive, r, 0.0)
            succ = succ | (alive & d & (r > 0.5))
            lava = lava | (alive & d & (r < -0.5))
            tend = jnp.where(alive & d, t + 1, tend)
            rec = (obs, a, r, alive) if record else None
            return (s2, o2, alive & ~d, ret + r, succ, lava, tend), rec

        f = jnp.zeros(B, bool)
        carry0 = (states, obs0, ~f, jnp.zeros(B), f, f, jnp.full(B, N_STEPS))
        (_, _, _, ret, succ, lava, tend), rec = jax.lax.scan(step, carry0, jnp.arange(N_STEPS))
        res = {"success": succ, "lava": lava, "ret": ret, "t_end": tend}
        if record:
            obs, a, r, al = rec
            res.update(obs=jnp.swapaxes(obs, 0, 1), act=jnp.swapaxes(a, 0, 1), rew=jnp.swapaxes(r, 0, 1),
                       alive=jnp.swapaxes(al, 0, 1))
        return res

    return jax.jit(run)
