"""sampling_mpc.py — Sampling-based MPC on the MuJoCo HalfCheetah.

Usage:
    python sampling_mpc.py --method mppi --K 256 --H 64 --n 8 --seed 0
    python sampling_mpc.py --method mppi --K 256 --H 64 --n 8 --seed 0 --visualize
    python sampling_mpc.py --sweep spline
    python sampling_mpc.py --sweep methods
    python sampling_mpc.py --sweep sigma_ramp
"""

import time
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import mujoco
import mujoco.rollout
import numpy as np
import tyro
import viser
from judo.visualizers.model import ViserMjModel

from solution import cem_update, mppi_update, ps_update, query_zoh, shift_warmstart

matplotlib.use("Agg")

# ── paths ──────────────────────────────────────────────────────────────────────
CHEETAH_XML = Path(__file__).parent / "data" / "xml" / "cheetah.xml"


# ── cost function (numpy, operates on rollout states) ─────────────────────────
# Cheetah state layout (mjSTATE_FULLPHYSICS, nstate=19):
#   [time(1) | qpos(9) | qvel(9)]
#   qvel[0] = rootx velocity = forward speed → index 10.


def cost_fn(states: np.ndarray, vel_scale: float = 10.0) -> np.ndarray:
    """Cheetah running cost. states: (K, H, nstate) → (K,)."""
    vel = np.clip(states[:, :, 10] / vel_scale, 0.0, 1.0)
    return np.sum(1.0 - vel, axis=1)


# ── per-step reward function (for cumulative-reward tracking in run_episode) ───


def _step_reward_cheetah(data_sim: mujoco.MjData, _model) -> float:
    return float(np.clip(data_sim.qvel[0] / 10.0, 0.0, 1.0))


# ── episode runner ─────────────────────────────────────────────────────────────


def run_episode(
    model: mujoco.MjModel,
    method: str,
    K: int,
    H: int,
    n: int,
    seed: int,
    cost_fn,
    step_reward_fn=_step_reward_cheetah,
    lam: float = 0.01,
    elite_frac: float = 0.1,
    init_sigma: float = 1.0,
    sigma_ramp: float = 1.0,
    T_max: int = 500,
    ctrl_steps: int = 8,
    record: bool = False,
) -> "float | tuple[float, list]":
    """Run one MPC episode and return cumulative reward.

    One MPC iteration:
      - Plans over H steps using K samples and n ZOH knots.
      - Applies mu[0] for ctrl_steps sim steps (fixed, independent of H/n).
      - Warm-starts by querying the ZOH spline at the new knot times after
        advancing ctrl_steps steps, extrapolating with 0 past the horizon.

    step_reward_fn(data_sim, model) → float is called every sim step to
    accumulate the reported cumulative reward.

    If record=True, returns (cumulative_reward, trajectory) where trajectory is
    a list of (qpos, qvel) arrays captured at each sim step.
    """
    assert H % n == 0
    seg_len = H // n  # ZOH segment length for rollout only
    nu = model.nu
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    ctrl_min = model.actuator_ctrlrange[:, 0]
    ctrl_max = model.actuator_ctrlrange[:, 1]

    rng = np.random.default_rng(seed)
    data_sim = mujoco.MjData(model)
    rollout_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data_sim)

    mu = np.zeros((n, nu))
    sigma = np.full((n, nu), init_sigma)  # CEM updates this; MPPI/PS keep it fixed
    sigma_ramp_vec = np.linspace(sigma_ramp / n, sigma_ramp, n)[:, None]

    state_buf = np.zeros(nstate)
    init_states = np.zeros((K, nstate))

    cumulative_reward = 0.0
    trajectory: list = []

    # build spline time arrays (constant across iterations)
    knot_times_np = np.arange(n) * seg_len  # (n,) integer step indices
    knot_times = jnp.array(
        knot_times_np, dtype=jnp.float32
    )  # JAX version for query_zoh
    rollout_times = jnp.arange(H, dtype=jnp.float32)  # (H,) query times

    # jit-compile query_zoh via vmap once
    expand_batch = jax.jit(
        jax.vmap(lambda zi: query_zoh(knot_times, zi, rollout_times))
    )

    for _ in range(T_max):
        # 1. Sample knots ~ N(mu, sigma²), clipped to [-1, 1]
        z = mu[None] + rng.standard_normal((K, n, nu)) * (sigma * sigma_ramp_vec)
        z = np.clip(z, -1.0, 1.0)

        # 2. Expand to (K, H, nu) via ZOH
        actions = np.array(expand_batch(jnp.array(z, dtype=jnp.float32)))

        # 3. Batch rollout from current state
        mujoco.mj_getState(
            model, data_sim, state_buf, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        init_states[:] = state_buf
        states, _ = mujoco.rollout.rollout(model, rollout_data, init_states, actions)

        # 4. Compute costs (K,)
        costs = cost_fn(states)

        # 5. Update mean (and sigma for CEM)
        z_jnp = jnp.array(z, dtype=jnp.float32)
        c_jnp = jnp.array(costs, dtype=jnp.float32)
        if method == "mppi":
            mu = np.array(mppi_update(z_jnp, c_jnp, lam=lam))
        elif method == "cem":
            mu_j, sigma_j = cem_update(z_jnp, c_jnp, elite_frac=elite_frac)
            mu = np.array(mu_j)
            sigma = np.array(sigma_j)
        elif method == "ps":
            mu = np.array(ps_update(z_jnp, c_jnp))
        else:
            raise ValueError(f"Unknown method: {method}")

        # 6. Apply first knot action for ctrl_steps sim steps
        action = np.clip(mu[0], ctrl_min, ctrl_max)
        for _ in range(ctrl_steps):
            data_sim.ctrl[:] = action
            mujoco.mj_step(model, data_sim)
            cumulative_reward += step_reward_fn(data_sim, model)
            if record:
                trajectory.append((data_sim.qpos.copy(), data_sim.qvel.copy()))

        # 7. Warm-start: re-query the ZOH spline at new knot positions after
        #    advancing ctrl_steps. Knots that shift past the horizon extrapolate
        #    with 0 (mu) or init_sigma (sigma).
        mu = shift_warmstart(knot_times_np, mu, ctrl_steps, fill_value=0.0)
        sigma = shift_warmstart(knot_times_np, sigma, ctrl_steps, fill_value=init_sigma)

    return (cumulative_reward, trajectory) if record else cumulative_reward


def interactive_mode(model: mujoco.MjModel, args) -> None:
    """Viser viewer with parameter sliders, a Run button, and trajectory playback."""
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")

    mj_spec = mujoco.MjSpec.from_file(str(CHEETAH_XML))
    viser_model = ViserMjModel(server, mj_spec)
    data = mujoco.MjData(model)

    # ── MPC parameters ────────────────────────────────────────────────────────
    with server.gui.add_folder("MPC Parameters"):
        method_dd = server.gui.add_dropdown(
            "Method", ["mppi", "cem", "ps"], initial_value=args.method
        )
        K_slider = server.gui.add_slider(
            "K (samples)", min=64, max=4096, step=64, initial_value=args.K
        )
        H_slider = server.gui.add_slider(
            "H (horizon)", min=8, max=256, step=8, initial_value=args.H
        )
        n_slider = server.gui.add_slider(
            "n (knots)", min=1, max=64, step=1, initial_value=args.n
        )
        sigma_slider = server.gui.add_slider(
            "Init. std (σ)", min=0.05, max=2.0, step=0.05, initial_value=1.0
        )
        sigma_ramp_slider = server.gui.add_slider(
            "σ ramp", min=1.0, max=25.0, step=0.1, initial_value=1.0
        )
        lam_slider = server.gui.add_slider(
            "λ (MPPI)", min=0.001, max=1.0, step=0.001, initial_value=args.lam
        )
        elite_slider = server.gui.add_slider(
            "elite_frac (CEM)",
            min=0.01,
            max=0.5,
            step=0.01,
            initial_value=args.elite_frac,
        )
        T_slider = server.gui.add_slider(
            "T_max", min=10, max=200, step=10, initial_value=min(args.T_max, 100)
        )
        seed_slider = server.gui.add_slider(
            "Seed", min=0, max=99, step=1, initial_value=args.seed
        )

    # ── run controls ──────────────────────────────────────────────────────────
    run_button = server.gui.add_button("Run Episode")
    status_text = server.gui.add_text(
        "Status", initial_value="Adjust params and click Run."
    )

    # ── playback controls ─────────────────────────────────────────────────────
    with server.gui.add_folder("Playback"):
        play_button = server.gui.add_button("Play / Replay")
        step_slider = server.gui.add_slider(
            "Step", min=0, max=1, step=1, initial_value=0
        )
        speed_slider = server.gui.add_slider(
            "Playback speed", min=0.1, max=4.0, step=0.1, initial_value=1.0
        )

    trajectory: list = []
    play_flag = False

    def _apply_frame(idx: int) -> None:
        qpos, qvel = trajectory[idx]
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        viser_model.set_data(data)
        step_slider.value = idx

    @run_button.on_click
    def _(_) -> None:
        nonlocal trajectory, play_flag
        play_flag = False

        K = int(K_slider.value)
        H = int(H_slider.value)
        n = int(n_slider.value)
        if H % n != 0:
            status_text.value = f"Error: H={H} must be divisible by n={n}."
            return

        status_text.value = "Running..."
        reward, traj = cast(
            tuple[float, list],
            run_episode(
                model,
                method=method_dd.value,
                K=K,
                H=H,
                n=n,
                seed=int(seed_slider.value),
                cost_fn=cost_fn,
                step_reward_fn=_step_reward_cheetah,
                lam=float(lam_slider.value),
                elite_frac=float(elite_slider.value),
                init_sigma=float(sigma_slider.value),
                sigma_ramp=float(sigma_ramp_slider.value),
                T_max=int(T_slider.value),
                record=True,
            ),
        )
        trajectory[:] = traj
        step_slider.max = len(traj) - 1
        step_slider.value = 0
        status_text.value = f"Done — cumulative reward: {reward:.2f}"
        _apply_frame(0)

    @play_button.on_click
    def _(_) -> None:
        nonlocal play_flag
        step_slider.value = 0
        play_flag = True

    print("Viser viewer ready. Open the URL above.")

    while True:
        if play_flag and trajectory:
            play_flag = False
            for i in range(len(trajectory)):
                _apply_frame(i)
                time.sleep(model.opt.timestep / float(speed_slider.value))
                if play_flag:
                    break
        elif trajectory:
            _apply_frame(int(step_slider.value))
            time.sleep(0.016)
        else:
            time.sleep(0.016)


# ── sweep helpers ──────────────────────────────────────────────────────────────


def run_grid(
    model,
    method,
    K,
    H,
    n_knots,
    seeds,
    cost_fn,
    lam,
    elite_frac,
    T_max,
    sigma_ramp=5.0,
):
    """Run M seeds and return array of cumulative rewards."""
    rewards = []
    for seed in seeds:
        r = run_episode(
            model,
            method,
            K,
            H,
            n_knots,
            seed,
            cost_fn,
            lam=lam,
            elite_frac=elite_frac,
            T_max=T_max,
            sigma_ramp=sigma_ramp,
        )
        rewards.append(r)
    return np.array(rewards)


def sweep_spline(model, M=5, T_max=25):
    """MPPI, K=256, H=64, n ∈ {1, 2, 4, 8, 16, 32, 64}, M seeds."""
    K, H = 256, 64
    ns = [1, 2, 4, 8, 16, 32, 64]
    seeds = list(range(M))
    results = {}
    for n in ns:
        print(f"  spline sweep: n={n}", flush=True)
        rewards = run_grid(
            model,
            "mppi",
            K,
            H,
            n,
            seeds,
            cost_fn,
            lam=0.01,
            elite_frac=0.05,
            T_max=T_max,
        )
        results[n] = rewards
    return results


def sweep_methods(model, M=5, T_max=25):
    """3 methods, n=8, H=64, K ∈ {32, 64, 256}, M seeds."""
    n, H = 8, 64
    Ks = [32, 64, 256]
    seeds = list(range(M))
    results = {}
    for method in ["mppi", "cem", "ps"]:
        results[method] = {}
        for K in Ks:
            print(f"  methods sweep: method={method} K={K}", flush=True)
            rewards = run_grid(
                model,
                method,
                K,
                H,
                n,
                seeds,
                cost_fn,
                lam=0.01,
                elite_frac=0.1,
                T_max=T_max,
            )
            results[method][K] = rewards
    return results


def sweep_sigma_ramp(model, M=5, T_max=25):
    """MPPI, K=256, H=64, n=8, sigma_ramp ∈ {0.5, 1.0, 2.5, 5.0, 7.5, 10.0}, M seeds."""
    K, H, n = 256, 64, 8
    sigma_ramps = [1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 25.0]
    seeds = list(range(M))
    results = {}
    for sr in sigma_ramps:
        print(f"  sigma_ramp sweep: sigma_ramp={sr}", flush=True)
        rewards = run_grid(
            model,
            "mppi",
            K,
            H,
            n,
            seeds,
            cost_fn,
            lam=0.01,
            elite_frac=0.1,
            T_max=T_max,
            sigma_ramp=sr,
        )
        results[sr] = rewards
    return results


# ── plotting ───────────────────────────────────────────────────────────────────


def _mean_stderr(arr):
    return arr.mean(), arr.std() / np.sqrt(len(arr))


def plot_spline(results, out_dir: Path):
    ns = sorted(results.keys())
    means = [_mean_stderr(results[n])[0] for n in ns]
    errs = [_mean_stderr(results[n])[1] for n in ns]
    fig, ax = plt.subplots()
    ax.errorbar(ns, means, yerr=errs, marker="o", capsize=4)
    ax.set_xlabel("Number of knots (n)")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Spline resolution: MPPI, K=256, H=64")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.tight_layout()
    path = out_dir / "sweep_spline.pdf"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def plot_methods(results, out_dir: Path):
    Ks = sorted(next(iter(results.values())).keys())
    fig, ax = plt.subplots()
    for method, data in results.items():
        means = [_mean_stderr(data[K])[0] for K in Ks]
        errs = [_mean_stderr(data[K])[1] for K in Ks]
        ax.errorbar(Ks, means, yerr=errs, marker="o", capsize=4, label=method)
    ax.set_xlabel("Number of samples (K)")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Methods: n=8, H=64, dense cost")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "sweep_methods.pdf"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def plot_sigma_ramp(results, out_dir: Path):
    sigma_ramps = sorted(results.keys())
    means = [_mean_stderr(results[sr])[0] for sr in sigma_ramps]
    errs = [_mean_stderr(results[sr])[1] for sr in sigma_ramps]
    fig, ax = plt.subplots()
    ax.errorbar(sigma_ramps, means, yerr=errs, marker="o", capsize=4)
    ax.set_xlabel("σ ramp")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Sigma ramp: MPPI, K=256, H=64, n=8")
    fig.tight_layout()
    path = out_dir / "sweep_sigma_ramp.pdf"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main(
    sweep: str | None = None,
    method: str = "mppi",
    K: int = 256,
    H: int = 64,
    n: int = 8,
    seed: int = 0,
    lam: float = 0.01,
    elite_frac: float = 0.1,
    T_max: int = 25,
    out_dir: Path = Path("plots"),
    seeds: int = 5,
    visualize: bool = False,
) -> None:
    """Sampling-based MPC on the MuJoCo HalfCheetah.

    Args:
        sweep: Run a parameter sweep ('spline', 'methods', or 'sigma_ramp').
        method: MPC method ('mppi', 'cem', or 'ps').
        K: Number of trajectory samples per iteration.
        H: Planning horizon (must be divisible by n).
        n: Number of ZOH spline knots.
        seed: Random seed for a single episode.
        lam: MPPI temperature parameter.
        elite_frac: Elite fraction for CEM.
        T_max: Number of MPC iterations per episode.
        out_dir: Directory for saving sweep plots.
        seeds: Number of seeds per config in sweeps.
        visualize: Open the interactive Viser viewer.
    """
    model = mujoco.MjModel.from_xml_path(str(CHEETAH_XML))
    out_dir.mkdir(parents=True, exist_ok=True)

    if sweep == "spline":
        print("Running spline sweep...")
        results = sweep_spline(model, M=seeds, T_max=T_max)
        plot_spline(results, out_dir)

    elif sweep == "methods":
        print("Running methods sweep...")
        results = sweep_methods(model, M=seeds, T_max=T_max)
        plot_methods(results, out_dir)

    elif sweep == "sigma_ramp":
        print("Running sigma_ramp sweep...")
        results = sweep_sigma_ramp(model, M=seeds, T_max=T_max)
        plot_sigma_ramp(results, out_dir)

    else:
        if visualize:
            interactive_mode(
                model,
                args=type(
                    "_Args",
                    (),
                    {
                        "method": method,
                        "K": K,
                        "H": H,
                        "n": n,
                        "lam": lam,
                        "elite_frac": elite_frac,
                        "T_max": T_max,
                        "seed": seed,
                    },
                )(),
            )
        else:
            reward = run_episode(
                model,
                method=method,
                K=K,
                H=H,
                n=n,
                seed=seed,
                cost_fn=cost_fn,
                lam=lam,
                elite_frac=elite_frac,
                T_max=T_max,
            )
            print(f"cumulative_reward={reward:.2f}")


if __name__ == "__main__":
    tyro.cli(main)
