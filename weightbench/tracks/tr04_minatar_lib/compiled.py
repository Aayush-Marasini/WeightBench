"""Hand-compiled reference policies for TR04-GAME-001..003 (no gradient descent, no data).

Each policy is a small exact ReLU circuit written unit by unit into the zero ``600→32→32→6``
shell. Observations are binary, so sums of indicator inputs are small integers, and
``relu(a + b - 1)`` is an exact AND of two indicators. Candidates differ only in one design
choice. The reference job picks the best one on private dev seeds (never on the eval seeds),
then rescales it to the Frobenius norm of a default-initialised network (function-preserving).
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from . import spec as S
from .spec import index

Weights = dict[str, np.ndarray]
Terms = dict[int, float]   # input (or hidden-unit) index -> weight


def _merge(*parts: Terms) -> Terms:
    out: Terms = {}
    for part in parts:
        for i, w in part.items():
            out[i] = out.get(i, 0.0) + w
    return out


def _scaled(terms: Terms, factor: float) -> Terms:
    return {i: factor * w for i, w in terms.items()}


def _row_cells(channel: int, row: int, weight: float = 1.0) -> Terms:
    return {index(channel, row, col): weight for col in range(S.GRID)}


def _column_sum(channel: int, rows: range = range(S.GRID)) -> Terms:
    """Σ col · x[channel, row, col]: the column of a single active cell in that channel."""
    return {index(channel, row, col): float(col) for row in rows for col in range(S.GRID)}


# ---------------------------------------------------------------- observation delay (knob level >= 1)
# Under delay d the input is ``[frame(t − d) ; d one-hot slots]`` (``spec.delayed_observation``): slot i holds the action
# the policy itself chose at t − d + i. The game source says each ``plus`` / ``minus`` action moves the agent one cell,
# so the agent's current coordinate is the shown coordinate plus Σ_i (slot_i[plus] − slot_i[minus]): a LINEAR read-out
# of the input ("dead reckoning"). It ignores sticky repeats (p = 0.1) and wall clipping; the circuits below never
# press an action that clips while their estimate is right, so an error lasts at most d steps (it leaves the window).
def _slot_shift(d: int, plus: str, minus: str) -> Terms:
    """Σ_i (slot_i[plus] − slot_i[minus]) over the d action slots: the agent's own displacement since the shown frame."""
    p, m = S.ACTIONS.index(plus), S.ACTIONS.index(minus)
    out: Terms = {}
    for i in range(d):
        out[S.N_IN + S.N_ACTIONS * i + p] = 1.0
        out[S.N_IN + S.N_ACTIONS * i + m] = -1.0
    return out


def _lane_decoder(net: "CircuitBuilder", channel: int, lo: int, hi: int,
                  d: int) -> Callable[[int], tuple[Terms, float]]:
    """Layer-1 units ``q_j = relu(cur − j)`` for j = lo..hi, where ``cur`` = shown row of the single ``channel`` cell plus
    the pending ``d`` − ``u`` slots. Returns ``lane(k) -> (terms over q units, constant)`` whose sum is exactly the
    indicator [current row == k] for integer ``cur``, saturating at the ends (row lo ⇐ cur ≤ lo, row hi ⇐ cur ≥ hi):
    ``[cur ≥ k] = q_{k−1} − q_k``, so [cur == k] is a second difference. Uses hi − lo + 1 units, as many as the
    one-hot row units it replaces."""
    cur = _merge({index(channel, r, c): float(r) for r in range(S.GRID) for c in range(S.GRID)},
                 _slot_shift(d, "d", "u"))
    q = {j: net.hidden1(cur, bias=-float(j)) for j in range(lo, hi + 1)}

    def lane(k: int) -> tuple[Terms, float]:
        if k == lo:
            return {q[lo]: -1.0, q[lo + 1]: 1.0}, 1.0
        if k == hi:
            return {q[hi - 1]: 1.0, q[hi]: -1.0}, 0.0
        return {q[k - 1]: 1.0, q[k]: -2.0, q[k + 1]: 1.0}, 0.0

    return lane


class CircuitBuilder:
    """Allocates hidden units one at a time in an all-zero network."""

    def __init__(self, d: int = 0) -> None:
        self.weights = S.zero_weights(S.param_shapes(d))
        self._n_hidden1 = 0
        self._n_hidden2 = 0

    def hidden1(self, terms: Terms, bias: float = 0.0) -> int:
        unit = self._next("1")
        for i, w in terms.items():
            self.weights["W1"][unit, i] = w
        self.weights["b1"][unit] = bias
        return unit

    def hidden2(self, terms: Terms, bias: float = 0.0) -> int:
        unit = self._next("2")
        for i, w in terms.items():
            self.weights["W2"][unit, i] = w
        self.weights["b2"][unit] = bias
        return unit

    def output(self, action: str, terms: Terms, bias: float = 0.0) -> None:
        a = S.ACTIONS.index(action)
        for i, w in terms.items():
            self.weights["W3"][a, i] = w
        self.weights["b3"][a] = bias

    def _next(self, layer: str) -> int:
        if layer == "1":
            unit, self._n_hidden1 = self._n_hidden1, self._n_hidden1 + 1
            limit = S.HIDDEN[0]
        else:
            unit, self._n_hidden2 = self._n_hidden2, self._n_hidden2 + 1
            limit = S.HIDDEN[1]
        if unit >= limit:
            raise ValueError(f"hidden layer {layer} has only {limit} units")
        return unit


# ---------------------------------------------------------------------------------- breakout
def breakout(predict: bool = True) -> Weights:
    """Move the paddle toward the ball's next column.

    Channels: 0 paddle (row 9 only), 1 ball, 2 trail (the ball's previous cell). With
    ``predict`` the target is ``2·ball_x − trail_x`` (one diagonal step ahead), else ``ball_x``.
    Layer 1 splits ``target − paddle_x`` into its positive and negative parts, layer 2 copies
    them, and the outputs are ``right = (target − paddle)+``, ``left = (paddle − target)+`` and
    ``no-op = 0.5``. Differences are integers, so any non-zero gap beats the no-op bias.
    """
    net = CircuitBuilder()
    paddle_x = {index(0, 9, col): float(col) for col in range(S.GRID)}
    if predict:
        target = _merge(_scaled(_column_sum(1), 2.0), _scaled(_column_sum(2), -1.0))
    else:
        target = _column_sum(1)
    gap = _merge(target, _scaled(paddle_x, -1.0))
    right1 = net.hidden1(gap)
    left1 = net.hidden1(_scaled(gap, -1.0))
    net.output("r", {net.hidden2({right1: 1.0}): 1.0})
    net.output("l", {net.hidden2({left1: 1.0}): 1.0})
    net.output("n", {}, bias=0.5)
    return net.weights


def ball_column_after(col: int, right: bool, n: int) -> int:
    """Breakout ball column ``n`` steps after column ``col`` moving right/left (``Env.act``): a wall clamps the new
    column and flips the direction, so the ball sits on the wall column for two frames (…, 8, 9, 9, 8, …). The
    horizontal motion is periodic with period 20; bricks, the ceiling and a centre paddle hit flip only the vertical
    direction (only a paddle-edge hit, in row 8 and long before the ball returns, flips it horizontally)."""
    u = (col + n if right else 19 - col + n) % 20
    return u if u < 10 else 19 - u


def breakout_delayed(d: int, ahead: int) -> Weights:
    """Breakout under observation delay ``d``: move the paddle toward the ball's column ``ahead`` steps after the
    shown frame (``ahead = d + 1`` is the L0 ``predict_next_column`` rule shifted by the delay).

    - Current paddle: ``P = Σ col·paddle[9, col] + Σ_i (slot_i[r] − slot_i[l])`` (``_slot_shift``).
    - Ball direction from the shown ball and trail cells: ``D = 2·(ball_x − trail_x) + ball[col 0] − ball[col 9]``.
      ``D ≥ 1`` means moving right (also the spawn and wall cases, where trail = ball), ``D ≤ −1`` left.
    - Targets: ``T± = Σ ball[row, c] · ball_column_after(c, ±, ahead)``, an exact lookup on the one-hot ball cell.

    Layer 1: ``(T+ − P)±``, ``(T− − P)±`` and ``relu(D), relu(D − 1), relu(−D), relu(−D − 1)``; the last four give the
    exact direction indicators ``[D ≥ 1] = relu(D) − relu(D − 1)`` and ``[D ≤ −1]`` in layer 2, which gate the four gap
    units (GATE exceeds every gap). Outputs as at L0: ``right``/``left`` = gated positive/negative gap, ``no-op`` = 0.5.
    """
    net = CircuitBuilder(d)
    paddle = _merge({index(0, 9, col): float(col) for col in range(S.GRID)}, _slot_shift(d, "r", "l"))
    gaps = {}
    for right in (True, False):
        target = {index(1, row, c): float(ball_column_after(c, right, ahead))
                  for row in range(S.GRID) for c in range(S.GRID)}
        gap = _merge(target, _scaled(paddle, -1.0))
        gaps[right] = (net.hidden1(gap), net.hidden1(_scaled(gap, -1.0)))
    direction = _merge(_scaled(_column_sum(1), 2.0), _scaled(_column_sum(2), -2.0),
                       {index(1, row, 0): 1.0 for row in range(S.GRID)},
                       {index(1, row, 9): -1.0 for row in range(S.GRID)})
    d_pos, d_pos1 = net.hidden1(direction), net.hidden1(direction, bias=-1.0)
    d_neg, d_neg1 = net.hidden1(_scaled(direction, -1.0)), net.hidden1(_scaled(direction, -1.0), bias=-1.0)
    is_dir = {True: {d_pos: 1.0, d_pos1: -1.0}, False: {d_neg: 1.0, d_neg1: -1.0}}
    move = {"r": {}, "l": {}}
    for right in (True, False):
        gate = _scaled(is_dir[right], GATE)
        for out, unit in zip(("r", "l"), gaps[right]):
            move[out][net.hidden2(_merge({unit: 1.0}, gate), bias=-GATE)] = 1.0
    net.output("r", move["r"])
    net.output("l", move["l"])
    net.output("n", {}, bias=0.5)
    return net.weights


# ------------------------------------------------------------------- Asterix and Seaquest shared
HOME_COL = 5     # both games start the agent in column 5; the circuits below never move it sideways
GATE = 100.0     # larger than any sum a gated layer-2 unit can see


def _closing(net: CircuitBuilder, row: int, target: int, other: int, trail: int,
             value: Callable[[int], float], k: float) -> int:
    """``value(c)`` if the single ``target`` entity of ``row`` (at column c) is closing on column 5, else 0.

    The trail cell sits behind the entity, so ``Σ |c − 5| · (trail − target)`` is +1 when it closes on the home
    column and −1 when it leaves (no trail yet counts as leaving). ``k`` weighs that sign against ``value``;
    any ``other`` entity in the row switches the unit off (its trail would corrupt the sign).
    """
    terms = {}
    for c in range(S.GRID):
        reach = abs(c - HOME_COL)
        terms[index(target, row, c)] = value(c) - k * reach
        terms[index(trail, row, c)] = k * reach
        terms[index(other, row, c)] = -5.0 * k
    return net.hidden1(terms, bias=-k)


# ----------------------------------------------------------------------------------- asterix
ASTERIX_ROWS = range(1, 9)   # entity lanes and the player's vertical range
LANE_GAIN = 4.0              # flee gain and neighbour-lane penalty (> any gold pull)


def asterix(window: int = 2, d: int = 0) -> Weights:
    """Hold column 5, hop between lanes: collect gold that is still coming, flee enemies that are coming.

    Channels: 0 player, 1 enemy, 2 trail, 3 gold. Each lane holds at most one entity. Layer 1:
    - ``player_r``: the player's lane;
    - ``threat_r``: ``window + 1 − |c − 5|`` for an enemy at column c that is closing on column 5 (0 beyond the window);
    - ``near_r`` / ``far_r``: an indicator for gold closing on column 5 within / beyond 2 columns.

    Layer 2, one unit per lane and direction, gated by ``player_r``:
    ``relu(LANE_GAIN·(threat_here − threat_next) + pull − 10·gold_here)``. Fleeing wins over seeking, a lane
    whose enemy is closing within the window is never entered, and the player waits in a lane whose gold is on
    its way. The pull of gold in a lane is ``near`` (1.0 / 0.7 for 1 / 2 lanes away) plus ``far`` (0.8 / distance).

    Under an observation delay ``d`` > 0 the gate ``player_r`` is replaced by the dead-reckoned current lane
    (``_lane_decoder``: the shown lane plus the pending ``d`` / ``u`` slots); everything else is unchanged.
    """
    net = CircuitBuilder(d)
    player, threat, near, far = {}, {}, {}, {}
    lane = _lane_decoder(net, 0, ASTERIX_ROWS.start, ASTERIX_ROWS.stop - 1, d) if d else None
    for r in ASTERIX_ROWS:
        if not d:
            player[r] = net.hidden1(_row_cells(0, r))
        threat[r] = _closing(net, r, 1, 3, 2, lambda c: max(0.0, window + 1 - abs(c - HOME_COL)), k=window + 2.0)
        near[r] = _closing(net, r, 3, 1, 2, lambda c: float(abs(c - HOME_COL) <= 2), k=2.0)
        far[r] = _closing(net, r, 3, 1, 2, lambda c: float(abs(c - HOME_COL) > 2), k=2.0)

    def lane_move(k: int, step: int) -> int:
        """Unit for moving from lane k to lane k + step (±1)."""
        pull = {}
        for r in ASTERIX_ROWS:
            gap = (r - k) * step
            if gap > 0:
                pull = _merge(pull, {near[r]: {1: 1.0, 2: 0.7}.get(gap, 0.0), far[r]: 0.8 / gap})
        gate, const = lane(k) if d else ({player[k]: 1.0}, 0.0)
        terms = _merge(pull, _scaled(gate, GATE), {threat[k]: LANE_GAIN, near[k]: -10.0, far[k]: -10.0,
                                                    threat[k + step]: -LANE_GAIN})
        return net.hidden2(terms, bias=GATE * const - GATE)

    net.output("u", {lane_move(k, -1): 1.0 for k in ASTERIX_ROWS if k - 1 in threat})
    net.output("d", {lane_move(k, +1): 0.99 for k in ASTERIX_ROWS if k + 1 in threat})   # ties go up
    net.output("n", {}, bias=0.02)
    return net.weights


# ---------------------------------------------------------------------------------- seaquest
SEAQUEST_ROWS = range(2, 9)   # patrolled rows; row 1 is only a transit row (a stray 'up' there would end the game)
# Hazard weights by column. TARGET: hazards left of the sub (it shoots left) are worth a visit. THREAT: hazards
# near the sub (column 5 only matters in the next row) must be dodged, and rows holding them are not entered.
TARGET = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.5}
THREAT = {3: 2.0, 4: 4.0, 5: 8.0, 6: 4.0, 7: 2.0}
FLEE, STAY, AIR = 4.0, 5.0, 10.0


def seaquest(home: int = 4, air: int = 3, diver: float = 1.5, d: int = 0) -> Weights:
    """Patrol rows 2..8 at column 5 facing left, shooting; hunt enemies and divers, dodge, surface for air.

    Channels: 0 sub front, 3 hazards (fish, enemy subs, enemy bullets), 4 divers + diver gauge (row 9),
    5 trails + oxygen gauge (row 9). The sub never turns, so everything on its left is shootable (an enemy that
    has passed it is shot from behind) and everything on its right is a threat.

    Layer 1: ``sub_k`` (the sub's row); ``target_r`` (weighted hazards left of the sub, see TARGET);
    ``diver_r`` (a diver closing on column 5); ``threat_r`` (see THREAT); ``need_air = relu(diver_gauge[9, 8] −
    oxygen_gauge[9, air])`` (a diver aboard and oxygen below 20·(air + 1) of 200).

    Layer 2, gated by ``sub_k``: ``move_k`` (up or down) = ``Σ`` pull of the targets in the rows that way (1/distance,
    ``diver``/distance for divers) − STAY·(targets in the own row: stay and shoot) + FLEE·(threat_here − threat_next)
    (dodge into a calmer row, never into a threatened one) + 0.3 toward row ``home``; ``surface_k`` = AIR·need_air −
    FLEE·threat_next. Output ``f`` has bias 0.1, so the sub shoots whenever nothing moves it.

    Under an observation delay ``d`` > 0 the gate ``sub_k`` is replaced by the dead-reckoned current row
    (``_lane_decoder``: the shown row plus the pending ``d`` / ``u`` slots); everything else is unchanged.
    """
    net = CircuitBuilder(d)
    if d:
        lane = _lane_decoder(net, 0, 0, S.GRID - 2, d)
    else:
        sub = {k: net.hidden1(_row_cells(0, k)) for k in range(S.GRID - 1)}

        def lane(k: int) -> tuple[Terms, float]:
            return {sub[k]: 1.0}, 0.0
    target = {r: net.hidden1({index(3, r, c): w for c, w in TARGET.items()}) for r in SEAQUEST_ROWS}
    closing_diver = {r: _closing(net, r, 4, 3, 5, lambda c: 1.0, k=2.0) for r in SEAQUEST_ROWS}
    threat = {r: net.hidden1({index(3, r, c): w for c, w in THREAT.items()}) for r in range(1, 9)}
    need_air = net.hidden1({index(4, 9, 8): 1.0, index(5, 9, air): -1.0})

    def move(k: int, step: int) -> int:
        """Unit for moving from row k to row k + step (-1 = up, +1 = down)."""
        pull = {}
        for r in SEAQUEST_ROWS:
            gap = (r - k) * step
            if gap > 0:
                pull = _merge(pull, {target[r]: 1.0 / gap, closing_diver[r]: diver / gap})
        stay = {target[k]: -STAY, closing_diver[k]: -STAY} if k in target else {}
        risk = {threat[k + step]: -FLEE, **({threat[k]: FLEE} if k in threat else {})}
        toward_home = 0.3 if (home - k) * step > 0 else 0.0
        gate, const = lane(k)
        return net.hidden2(_merge(pull, stay, risk, _scaled(gate, GATE)), bias=toward_home + GATE * const - GATE)

    def surface(k: int) -> int:
        gate, const = lane(k)
        terms = {**_scaled(gate, GATE), need_air: AIR}
        if k - 1 in threat:
            terms[threat[k - 1]] = -FLEE
        return net.hidden2(terms, bias=GATE * const - GATE)

    net.output("u", {**{move(k, -1): 1.0 for k in range(3, 9)}, **{surface(k): 1.0 for k in range(1, 9)}})
    net.output("d", {move(k, +1): 0.99 for k in range(0, 8)})   # ties go up
    net.output("f", {}, bias=0.1)
    return net.weights


# ------------------------------------------------------------------------------- candidates
CANDIDATES: dict[str, dict[str, Callable[[], Weights]]] = {
    "breakout": {"predict_next_column": lambda: breakout(predict=True),
                 "track_ball_column": lambda: breakout(predict=False)},
    "asterix": {f"window_{w}": (lambda w=w: asterix(w)) for w in (1, 2, 3)},
    "seaquest": {f"home_{h}_diver_{d}": (lambda h=h, d=d: seaquest(h, diver=d)) for h in (3, 4, 5) for d in (1.0, 2.5)},
}


def delayed_candidates(game: str, d: int) -> dict[str, Callable[[], Weights]]:
    """Reference candidates at observation delay ``d`` ≥ 1 (knob level ≥ 1): the L0 circuits with the agent's own
    position dead-reckoned from the action slots. Breakout varies the ball look-ahead (d, d + 1, d + 2 steps after the
    shown frame); Asterix widens the enemy window by up to d columns (entities keep moving during the delay); Seaquest
    keeps the L0 grid."""
    if d < 1:
        raise ValueError("delayed_candidates needs d >= 1")
    if game == "breakout":
        return {f"dead_reckon_ahead_{n}": (lambda n=n: breakout_delayed(d, n)) for n in (d, d + 1, d + 2)}
    if game == "asterix":
        return {f"dead_reckon_window_{w}": (lambda w=w: asterix(w, d=d)) for w in range(1, min(3 + d, 5) + 1)}
    if game == "seaquest":
        return {f"dead_reckon_home_{h}_diver_{v}": (lambda h=h, v=v: seaquest(h, diver=v, d=d))
                for h in (3, 4, 5) for v in (1.0, 2.5)}
    raise ValueError(f"unknown game {game!r}")


def scale_layers(weights: Mapping[str, np.ndarray], k: float) -> Weights:
    """Scale W_l by k and b_l by k**l (l = 1, 2, 3). Layer-l pre-activations scale by k**l, so
    for k > 0 the argmax policy is unchanged (positive homogeneity of ReLU)."""
    out = {}
    for name, w in weights.items():
        layer = int(name[1])
        power = layer if name.startswith("b") else 1
        out[name] = np.asarray(w, np.float64) * k ** power
    return out


def rescale(weights: Mapping[str, np.ndarray], target_norm: float) -> Weights:
    """Function-preserving rescale so the total Frobenius norm equals ``target_norm``."""
    lo, hi = 1e-6, 1e6
    for _ in range(200):   # the norm is increasing in k: bisect in log space
        mid = float(np.sqrt(lo * hi))
        if S.frobenius_norm(scale_layers(weights, mid)) < target_norm:
            lo = mid
        else:
            hi = mid
    k = float(np.sqrt(lo * hi))
    return {name: w.astype(np.float32) for name, w in scale_layers(weights, k).items()}
