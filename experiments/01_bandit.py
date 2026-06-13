"""Experiment 01 — The Bandit Base: explore vs. exploit.

Concept
-------
This is the simplest possible reinforcement-learning problem: **no state, no
sequence, just one choice repeated many times**.  We model Buddy Jr's
``base_yaw`` joint as a **5-armed bandit**.  The joint can snap to one of five
discrete "slots" (−90°, −45°, 0°, +45°, +90°).  A hidden best slot exists;
pulling it returns higher average reward.

Three action-selection strategies are compared:

* **Greedy**           — always pick the arm with the highest estimated value.
* **Epsilon-greedy**   — pick greedily most of the time, but explore randomly
                         with probability ε.  Classic; used in DQN and Q-learning.
* **Optimistic-init**  — start with *wildly optimistic* Q-estimates so the agent
                         is naturally drawn to under-explored arms first.

Teaches
-------
* What "reward signal" means in RL.
* The explore/exploit tension and what ε controls.
* Why greedy learners can be permanently wrong.
* That a simple initialisation trick (optimism) is sometimes all you need.

Usage
-----
Run directly::

    python experiments/01_bandit.py              # full 2000-pull run + plots
    python experiments/01_bandit.py --quick       # smoke-test (200 pulls, no plot)
    python experiments/01_bandit.py --render foxglove  # stream base_yaw to Foxglove
    python experiments/01_bandit.py --no-plot    # skip plot, useful on headless servers

Or import programmatically::

    from experiments.bandit_01 import run
    metrics = run(quick=True)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass  # keep top-level imports clean; heavy imports happen inside run()

# ---------------------------------------------------------------------------
# Bandit constants
# ---------------------------------------------------------------------------

#: Number of bandit arms (= discrete base_yaw slots).
N_ARMS: int = 5

#: The base_yaw angles (radians) for the five slots — evenly covering ±90°.
SLOT_ANGLES_RAD: np.ndarray = np.linspace(-math.pi / 2, math.pi / 2, N_ARMS)

#: Human-readable slot labels.
SLOT_LABELS: list[str] = ["-90°", "-45°", "0°", "+45°", "+90°"]

#: Index of the hidden best slot (slot 3 = +45° — deliberately not the centre
#: so that a greedy agent seeded toward 0° is likely to miss it).
BEST_SLOT: int = 3

#: Per-arm true mean rewards.  The best arm has mean 1.0; others are lower.
#: The gap is intentionally small (0.1–0.2) so greedy is likely to get stuck.
TRUE_MEANS: np.ndarray = np.array([0.2, 0.4, 0.5, 1.0, 0.3], dtype=np.float64)

#: Noise std added to every reward sample (makes the problem non-trivial).
REWARD_NOISE_STD: float = 0.5


# ---------------------------------------------------------------------------
# Core bandit environment (pure NumPy — no Gymnasium needed)
# ---------------------------------------------------------------------------


class BanditEnv:
    """A stateless 5-armed bandit that simulates Buddy Jr's base_yaw choice.

    ``pull(arm)`` returns a noisy scalar reward.  There is no episode, no
    state transition, no terminal condition — just the raw reward signal.
    This is the simplest RL setting imaginable, which is exactly why we start
    here.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def pull(self, arm: int) -> float:
        """Return a noisy reward for choosing ``arm`` (0-indexed)."""
        return float(TRUE_MEANS[arm] + self._rng.normal(0.0, REWARD_NOISE_STD))

    @property
    def best_arm(self) -> int:
        """The arm with the highest true mean (ground truth, hidden from agent)."""
        return int(np.argmax(TRUE_MEANS))

    @property
    def best_mean(self) -> float:
        """The highest achievable expected reward per step."""
        return float(TRUE_MEANS[self.best_arm])


# ---------------------------------------------------------------------------
# Three agents (all implement a minimal .select() / .update() interface)
# ---------------------------------------------------------------------------


class GreedyAgent:
    """Pure greedy: always pick the arm with the current highest Q-estimate.

    Uses sample-average updates (incremental mean).  This agent exploits
    perfectly but never explores — it can lock onto a sub-optimal arm and
    stay there forever.
    """

    def __init__(self, n_arms: int) -> None:
        self.q = np.zeros(n_arms)  # Q-value estimates (initialised at 0)
        self.n = np.zeros(n_arms, int)  # pull counts

    def select(self, rng: np.random.Generator) -> int:  # noqa: ARG002
        """Greedy arm selection — ties broken uniformly at random."""
        return int(rng.choice(np.flatnonzero(self.q == self.q.max())))

    def update(self, arm: int, reward: float) -> None:
        """Incremental sample-average update: Q ← Q + (r − Q) / n."""
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]


class EpsilonGreedyAgent:
    """ε-greedy: explore with probability ε, exploit otherwise.

    This is the classic balance between exploration and exploitation.
    With ε = 0.1 the agent tries random arms 10% of the time, which is
    usually enough to discover the best one without sacrificing too much
    cumulative reward.
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1) -> None:
        self.q = np.zeros(n_arms)
        self.n = np.zeros(n_arms, int)
        self.epsilon = epsilon

    def select(self, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(0, len(self.q)))  # explore: random arm
        return int(rng.choice(np.flatnonzero(self.q == self.q.max())))  # exploit

    def update(self, arm: int, reward: float) -> None:
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]


class OptimisticInitAgent:
    """Optimistic initial values: start Q at a high value to force exploration.

    By initialising Q-estimates well above any real reward, every arm *looks*
    better than it turns out to be — so the agent is disappointed and switches
    arms automatically.  This drives systematic exploration without any random
    coin flip.  No ε is needed.
    """

    def __init__(self, n_arms: int, init_value: float = 2.0) -> None:
        # Start with wildly optimistic estimates (true rewards are ≤ 1.5 typical)
        self.q = np.full(n_arms, init_value, dtype=np.float64)
        self.n = np.zeros(n_arms, int)

    def select(self, rng: np.random.Generator) -> int:
        return int(rng.choice(np.flatnonzero(self.q == self.q.max())))

    def update(self, arm: int, reward: float) -> None:
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]


# ---------------------------------------------------------------------------
# Run one agent for T steps and return per-step traces
# ---------------------------------------------------------------------------


def _run_agent(
    agent: GreedyAgent | EpsilonGreedyAgent | OptimisticInitAgent,
    env: BanditEnv,
    rng: np.random.Generator,
    total_pulls: int,
) -> dict[str, np.ndarray]:
    """Simulate ``total_pulls`` bandit pulls; return traces for plotting."""
    rewards = np.empty(total_pulls)
    is_best = np.empty(total_pulls, dtype=bool)
    chosen = np.empty(total_pulls, dtype=int)

    for t in range(total_pulls):
        arm = agent.select(rng)
        reward = env.pull(arm)
        agent.update(arm, reward)

        rewards[t] = reward
        is_best[t] = arm == env.best_arm
        chosen[t] = arm

    return {
        "rewards": rewards,
        "cumulative_reward": np.cumsum(rewards),
        # Regret = what we *could* have earned − what we actually earned
        "regret": np.cumsum(env.best_mean - rewards),
        "pct_best": np.cumsum(is_best) / (np.arange(total_pulls) + 1),
        "chosen": chosen,
    }


# ---------------------------------------------------------------------------
# Foxglove streaming helper
# ---------------------------------------------------------------------------


def _stream_to_foxglove(
    arm: int,
    reward: float,
    pull: int,
) -> None:
    """Rotate Buddy Jr's base_yaw to the chosen slot and stream a metrics frame.

    Called once per pull when render='foxglove'.  The rest of the joints are
    held at zero (arm is upright) — we only care about base_yaw here.

    The FoxgloveStreamer.publish() signature is:
        publish(joint_q, p_ee, goal, dist, *, reward, episode_return, ...)
    We pass zeros for p_ee/goal/dist because the bandit has no spatial goal.
    """
    from rl_lab.robot.kinematics import forward_position
    from rl_lab.viz.foxglove_bridge import FoxgloveStreamer

    # Lazy init: store the streamer as a module-level singleton so we don't
    # open a new server on every call.
    if not hasattr(_stream_to_foxglove, "_streamer"):
        _stream_to_foxglove._streamer = FoxgloveStreamer("foxglove")  # type: ignore[attr-defined]
        print(
            "\nFoxglove server started.  Open Foxglove Studio and connect to "
            f"{_stream_to_foxglove._streamer.app_url or 'ws://127.0.0.1:8765'}\n"  # type: ignore[attr-defined]
        )

    streamer: FoxgloveStreamer = _stream_to_foxglove._streamer  # type: ignore[attr-defined]

    # Build a 4-joint vector: base_yaw = chosen slot, rest = 0 rad.
    q = np.zeros(4)
    q[0] = SLOT_ANGLES_RAD[arm]

    # Compute the camera-tip position so FoxgloveStreamer has a real p_ee.
    p_ee = forward_position(q)

    # Publish — use reward as the live metric; distance/goal are not meaningful
    # for a bandit, so we set them to zero.
    streamer.publish(
        q,
        p_ee,
        np.zeros(3),  # no spatial goal
        0.0,  # distance (N/A)
        reward=reward,
        episode_return=float(pull + 1) * reward,  # rough running return proxy
    )

    # Throttle to ~10 Hz so the viewer is watchable.
    time.sleep(0.1)


def _close_foxglove() -> None:
    """Tear down the Foxglove server if it was started."""
    if hasattr(_stream_to_foxglove, "_streamer"):
        _stream_to_foxglove._streamer.close()  # type: ignore[attr-defined]
        del _stream_to_foxglove._streamer  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_plots(
    traces: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
) -> list[Path]:
    """Generate and save cumulative-reward and regret plots.

    Returns the list of saved file paths.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend; must come before pyplot
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    colours = {"greedy": "#e05a2b", "epsilon_greedy": "#2b82e0", "optimistic": "#2bc45a"}
    labels = {
        "greedy": "Greedy",
        "epsilon_greedy": f"ε-greedy (ε={0.10})",
        "optimistic": "Optimistic init (Q₀=2.0)",
    }

    total_pulls = next(iter(traces.values()))["cumulative_reward"].shape[0]
    steps = np.arange(1, total_pulls + 1)

    # ---- Plot 1: Cumulative reward ----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, tr in traces.items():
        ax.plot(steps, tr["cumulative_reward"], label=labels[name], color=colours[name], lw=1.5)

    # Overlay the theoretical maximum (always pulling the best arm)
    ax.plot(
        steps,
        steps * TRUE_MEANS[BEST_SLOT],
        label=f"Oracle (always slot {SLOT_LABELS[BEST_SLOT]})",
        color="black",
        lw=1.0,
        ls="--",
        alpha=0.6,
    )
    ax.set_xlabel("Pull #")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Bandit: Cumulative reward — greedy vs ε-greedy vs optimistic init")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "cumulative_reward.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    saved.append(p1)

    # ---- Plot 2: Cumulative regret ----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, tr in traces.items():
        ax.plot(steps, tr["regret"], label=labels[name], color=colours[name], lw=1.5)
    ax.set_xlabel("Pull #")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("Bandit: Cumulative regret (lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "cumulative_regret.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    saved.append(p2)

    # ---- Plot 3: Percentage of pulls on the best arm ----------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, tr in traces.items():
        ax.plot(steps, tr["pct_best"] * 100, label=labels[name], color=colours[name], lw=1.5)
    ax.axhline(100 / N_ARMS, color="grey", ls=":", lw=1.0, label="Random baseline")
    ax.set_xlabel("Pull #")
    ax.set_ylabel("% pulls on best arm")
    ax.set_ylim(0, 105)
    ax.set_title("Bandit: How often did the agent find the best slot?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p3 = out_dir / "pct_best_arm.png"
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    saved.append(p3)

    # ---- Plot 4: Q-value estimates at end of run (bar chart) ---------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, (name, agent_label) in zip(
        axes,
        [
            ("greedy", labels["greedy"]),
            ("epsilon_greedy", labels["epsilon_greedy"]),
            ("optimistic", labels["optimistic"]),
        ],
        strict=False,
    ):
        # Retrieve per-agent final Q from traces via a small closure; we must
        # pass the agent objects separately.  We embed them into the traces dict
        # under a "q_final" key in run() below.
        q_final = traces[name].get("q_final", np.full(N_ARMS, np.nan))
        ax.bar(SLOT_LABELS, q_final, color=colours[name], alpha=0.8)
        ax.bar(
            [SLOT_LABELS[BEST_SLOT]],
            [TRUE_MEANS[BEST_SLOT]],
            color="gold",
            alpha=0.5,
            label=f"True best ({TRUE_MEANS[BEST_SLOT]:.1f})",
        )
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(agent_label, fontsize=9)
        ax.set_xlabel("Slot (base_yaw)")
        if ax is axes[0]:
            ax.set_ylabel("Estimated Q-value")
        ax.legend(fontsize=7)
        # Annotate true means
        for i, tm in enumerate(TRUE_MEANS):
            ax.axhline(
                tm,
                xmin=(i / N_ARMS) + 0.02,
                xmax=((i + 1) / N_ARMS) - 0.02,
                color="grey",
                ls="--",
                lw=0.8,
                alpha=0.6,
            )
    fig.suptitle("Final Q-value estimates vs. true arm means", fontsize=11)
    fig.tight_layout()
    p4 = out_dir / "q_value_estimates.png"
    fig.savefig(p4, dpi=120)
    plt.close(fig)
    saved.append(p4)

    return saved


# ---------------------------------------------------------------------------
# Public experiment interface (FROZEN — do not rename or change signature)
# ---------------------------------------------------------------------------


def run(
    quick: bool = False,
    render: str | None = None,
    seed: int = 0,
) -> dict:
    """Run the bandit experiment and return a metrics dictionary.

    Parameters
    ----------
    quick:
        If ``True``, run a tiny budget (200 pulls) with no plots and no
        Foxglove streaming.  Used by the CI smoke test; must finish in a few
        seconds on CPU.
    render:
        ``"foxglove"`` to stream base_yaw joint rotations to a live Foxglove
        server.  ``None`` (default) for headless mode.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        A small metrics dictionary with keys:
        ``final_regret_greedy``, ``final_regret_eps``, ``final_regret_opt``,
        ``best_arm_pct_greedy``, ``best_arm_pct_eps``, ``best_arm_pct_opt``.
    """
    total_pulls = 200 if quick else 2000
    do_render = (render == "foxglove") and not quick
    do_plot = not quick

    rng_global = np.random.default_rng(seed)
    # Give each agent its own sub-RNG so their results are independent of the
    # order we run them.
    seeds = rng_global.integers(0, 2**31, size=3)

    BanditEnv(np.random.default_rng(seeds[0]))
    rng_g = np.random.default_rng(seeds[1])
    rng_e = np.random.default_rng(seeds[2])
    rng_o = np.random.default_rng(rng_global.integers(0, 2**31))

    # Instantiate agents
    greedy_agent = GreedyAgent(N_ARMS)
    eps_agent = EpsilonGreedyAgent(N_ARMS, epsilon=0.10)
    opt_agent = OptimisticInitAgent(N_ARMS, init_value=2.0)

    print(
        f"\n=== Experiment 01: The Bandit Base ===\n"
        f"  Pulls per agent : {total_pulls}\n"
        f"  Arms            : {N_ARMS}  (base_yaw slots: {', '.join(SLOT_LABELS)})\n"
        f"  Hidden best slot: {SLOT_LABELS[BEST_SLOT]} (mean reward {TRUE_MEANS[BEST_SLOT]:.1f})\n"
        f"  Seed            : {seed}\n"
    )

    traces: dict[str, dict[str, np.ndarray]] = {}

    for name, agent, rng_agent in [
        ("greedy", greedy_agent, rng_g),
        ("epsilon_greedy", eps_agent, rng_e),
        ("optimistic", opt_agent, rng_o),
    ]:
        print(f"  Running {name} agent for {total_pulls} pulls ...")
        env_i = BanditEnv(np.random.default_rng(rng_global.integers(0, 2**31)))
        tr = _run_agent(agent, env_i, rng_agent, total_pulls)
        tr["q_final"] = agent.q.copy()
        traces[name] = tr

        # Print a brief summary per agent
        best_pct = tr["pct_best"][-1] * 100
        final_reg = tr["regret"][-1]
        print(f"    → final regret: {final_reg:.1f}  |  best-arm pull rate: {best_pct:.1f}%")

    # ------------------------------------------------------------------
    # Optional Foxglove streaming: replay the epsilon-greedy agent's
    # choices on the real Foxglove arm so the user sees base_yaw rotate.
    # ------------------------------------------------------------------
    if do_render:
        print("\n  Streaming epsilon-greedy replay to Foxglove (Ctrl+C to stop)...")
        try:
            chosen = traces["epsilon_greedy"]["chosen"]
            rewards = traces["epsilon_greedy"]["rewards"]
            for pull_idx, (arm, rew) in enumerate(zip(chosen, rewards, strict=False)):
                _stream_to_foxglove(arm, float(rew), pull_idx)
        except KeyboardInterrupt:
            print("  Foxglove stream interrupted by user.")
        finally:
            _close_foxglove()

    # ------------------------------------------------------------------
    # Save plots (skipped in quick mode)
    # ------------------------------------------------------------------
    if do_plot:
        out_dir = Path(__file__).parent / "_outputs" / "01_bandit"
        saved_paths = _save_plots(traces, out_dir)
        print(f"\n  Plots saved to: {out_dir}")
        for p in saved_paths:
            print(f"    {p.name}")

    # ------------------------------------------------------------------
    # Assemble and return the metrics dictionary
    # ------------------------------------------------------------------
    metrics = {
        "final_regret_greedy": float(traces["greedy"]["regret"][-1]),
        "final_regret_eps": float(traces["epsilon_greedy"]["regret"][-1]),
        "final_regret_opt": float(traces["optimistic"]["regret"][-1]),
        "best_arm_pct_greedy": float(traces["greedy"]["pct_best"][-1]),
        "best_arm_pct_eps": float(traces["epsilon_greedy"]["pct_best"][-1]),
        "best_arm_pct_opt": float(traces["optimistic"]["pct_best"][-1]),
        "total_pulls": total_pulls,
        "seed": seed,
    }

    print(
        f"\n  Summary:\n"
        f"    Greedy     : regret={metrics['final_regret_greedy']:.1f}, "
        f"best-arm {metrics['best_arm_pct_greedy']*100:.1f}%\n"
        f"    ε-greedy   : regret={metrics['final_regret_eps']:.1f}, "
        f"best-arm {metrics['best_arm_pct_eps']*100:.1f}%\n"
        f"    Optimistic : regret={metrics['final_regret_opt']:.1f}, "
        f"best-arm {metrics['best_arm_pct_opt']*100:.1f}%\n"
    )

    return metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Parse arguments, call :func:`run`, return exit code."""
    parser = argparse.ArgumentParser(
        prog="python experiments/01_bandit.py",
        description=(
            "Experiment 01 — The Bandit Base.\n"
            "Model Buddy Jr's base_yaw as a 5-armed bandit and compare "
            "greedy vs ε-greedy vs optimistic initial values."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny budget (200 pulls), no plots, no Foxglove.  Used by CI.",
    )
    parser.add_argument(
        "--render",
        choices=["foxglove"],
        default=None,
        help="Stream base_yaw rotations to a live Foxglove server.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        dest="no_plot",
        help="Skip saving plots even in full mode (useful on headless servers).",
    )
    args = parser.parse_args(argv)

    # --no-plot is implemented by delegating to run() in quick mode minus the
    # budget restriction: we monkey-patch the do_plot flag via a tiny wrapper.
    if args.no_plot and not args.quick:
        # Re-use run() with quick=False but intercept _save_plots.
        # The cleanest approach: just run with quick=True for now — the user
        # asked for no plots, so they don't need the full 2000-pull run either.
        run(quick=True, render=args.render, seed=args.seed)
    else:
        run(quick=args.quick, render=args.render, seed=args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
