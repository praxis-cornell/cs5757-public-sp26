"""Trajectory tracking: compare Newton, Gradient Descent, and Gauss-Newton side by side.

All solvers use parallel line search. After convergence (or early stopping),
play back all three trajectories simultaneously.
"""

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

# Target: circle in the x-y plane at z=0.4, centered at (0.5, 0).
t = jnp.linspace(0, 2 * jnp.pi, 100)
target_traj = jnp.stack(
    [
        0.3 * jnp.cos(t) + 0.5,
        0.3 * jnp.sin(t),
        0.4 * jnp.ones_like(t),
    ],
    axis=-1,
)
T = target_traj.shape[0]

# Line search: 8 log-spaced step sizes from 1e-4 to 10.
ALPHAS = jnp.logspace(-4, 1, 8)

# Viser server (reuse if running in a notebook).
if "server" not in globals() or server is None:
    server = viser.ViserServer()
else:
    server.scene.reset()
    server.gui.reset()

server.scene.add_grid("/ground", width=2, height=2)

# Robot.
urdf = load_robot_description("panda_description")
target_link_name = "panda_hand"
robot = pk.Robot.from_urdf(urdf)

PANDA_HOME = jnp.array(
    [0, -jnp.pi / 4, 0, -3 * jnp.pi / 4, 0, jnp.pi / 2, jnp.pi / 4, 0.0]
)
N_JOINTS = PANDA_HOME.shape[0]
target_link_index = robot.links.names.index(target_link_name)
fk_ee = jaxlie.SE3(robot.forward_kinematics(PANDA_HOME)[target_link_index])


# --- Residuals and loss --------------------------------------------------- #


def unpack_joints(q: jnp.ndarray) -> jnp.ndarray:
    """Reshape flat decision variable (T*n_joints,) -> (T, n_joints)."""
    return q.reshape(-1, robot.joints.lower_limits.shape[0])


def form_residual(q: jnp.ndarray, ee_target: jaxlie.SE3) -> jnp.ndarray:
    """Concatenated residual vector for the full trajectory.

    Components (in order):
      - SE3 tracking error per frame (position weighted 1.0, rotation 0.5)
      - Joint limit violations (weighted 1e3)
      - Finite-difference joint velocities (weighted 1e-2, smoothness prior)
    """

    def tracking_residual(q_i: jnp.ndarray, tgt_i: jaxlie.SE3) -> jnp.ndarray:
        ee = jaxlie.SE3(robot.forward_kinematics(q_i)[target_link_index])
        err = jaxlie.manifold.rminus(tgt_i, ee)  # 6D log-map error
        return jnp.concatenate([err[..., :3], 0.5 * err[..., 3:]])

    qu = unpack_joints(q)
    tracking = jax.vmap(tracking_residual)(qu, ee_target).flatten()

    lower_viol = jnp.maximum(0, jnp.array(robot.joints.lower_limits) - qu).flatten()
    upper_viol = jnp.maximum(0, qu - jnp.array(robot.joints.upper_limits)).flatten()

    velocity = (qu[1:] - qu[:-1]).flatten()

    return jnp.concatenate(
        [tracking, 1e3 * lower_viol, 1e3 * upper_viol, 1e-2 * velocity]
    )


def loss(q: jnp.ndarray, ee_target: jaxlie.SE3) -> jnp.ndarray:
    """Scalar cost: 0.5 * ||r(q)||^2."""
    r = form_residual(q, ee_target)
    return 0.5 * jnp.dot(r, r)


# --- Solvers -------------------------------------------------------------- #
#
# Each solver computes a descent direction from the current iterate q.
# All three are then fed through the same parallel line search.


@jax.jit
def compute_newton_direction(q: jnp.ndarray, ee_target: jaxlie.SE3) -> jnp.ndarray:
    """Full Newton step: d = -H^{-1} g.

    Uses the exact Hessian of the scalar loss. A Levenberg-style diagonal
    regularizer (1e-1 * |diag(H)| + 1e-6) is added for numerical stability
    near saddle points or rank-deficient configurations.
    """
    grad = jax.grad(loss)(q, ee_target)
    hess = jax.hessian(loss)(q, ee_target)
    hess = hess + jnp.abs(jnp.diag(jnp.diag(hess) * 1e-1 + 1e-6))
    return -jnp.linalg.solve(hess, grad)


@jax.jit
def compute_grad_direction(q: jnp.ndarray, ee_target: jaxlie.SE3) -> jnp.ndarray:
    """Steepest descent: d = -g.

    No curvature information — relies entirely on line search for step sizing.
    Slow but guaranteed descent.
    """
    return -jax.grad(loss)(q, ee_target)


@jax.jit
def compute_gauss_newton_direction(
    q: jnp.ndarray, ee_target: jaxlie.SE3, lambda_: float = 1e-6
) -> jnp.ndarray:
    """Gauss-Newton step via matrix-free conjugate gradient.

    Solves (J^T J + λI) d = -J^T r without forming J explicitly:
      - J·v is computed via jvp (forward-mode)
      - J^T·u is computed via vjp (reverse-mode)
    The CG solve avoids materializing the (T*n_joints)^2 approximate Hessian.
    """

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        """Compute (J^T J + λI) v."""
        _, vjp_fn = jax.vjp(lambda q: form_residual(q, ee_target), q)
        Jv = jax.jvp(lambda q: form_residual(q, ee_target), [q], [v])[1]
        return vjp_fn(Jv)[0] + lambda_ * v

    _, vjp_fn = jax.vjp(lambda q: form_residual(q, ee_target), q)
    JTr = vjp_fn(form_residual(q, ee_target))[0]
    return jax.scipy.sparse.linalg.cg(matvec, -JTr, maxiter=20)[0]


@jax.jit
def line_search(
    q: jnp.ndarray, direction: jnp.ndarray, ee_target: jaxlie.SE3
) -> jnp.ndarray:
    """Parallel line search over ALPHAS. Returns the candidate with lowest loss."""
    candidates = q + ALPHAS[:, None] * direction
    losses = jax.vmap(loss, in_axes=(0, None))(candidates, ee_target)
    return candidates[jnp.argmin(losses)]


DIRECTION_FNS: dict[str, callable] = {
    "Newton": compute_newton_direction,
    "Gradient Descent": compute_grad_direction,
    "Gauss-Newton": compute_gauss_newton_direction,
}

# Color per solver: (R, G, B) in [0, 255].
SOLVER_COLORS: dict[str, tuple[int, int, int]] = {
    "Newton": (230, 80, 80),
    "Gradient Descent": (80, 180, 80),
    "Gauss-Newton": (80, 120, 230),
}


# --- Scene ---------------------------------------------------------------- #

# One semi-transparent robot per solver.
solver_robots: dict[str, ViserUrdf] = {}
for name, (r, g, b) in SOLVER_COLORS.items():
    key = name.replace(" ", "_").lower()
    solver_robots[name] = ViserUrdf(
        server,
        urdf,
        root_node_name=f"/{key}",
        mesh_color_override=[r / 255, g / 255, b / 255, 0.5],
    )
    solver_robots[name].update_cfg(onp.asarray(PANDA_HOME))

# Yellow target trajectory.
pts = onp.asarray(target_traj)
server.scene.add_line_segments(
    "/target_trail",
    points=onp.stack([pts[:-1], pts[1:]], axis=1),
    colors=(255, 200, 50),
    line_width=2.0,
)


# --- GUI ------------------------------------------------------------------ #

with server.gui.add_folder("Optimization"):
    gui_pause = server.gui.add_button("Pause / Resume")
    gui_step = server.gui.add_button("Step Once")
    gui_reset = server.gui.add_button("Reset All")
    gui_max_iters = server.gui.add_slider(
        "Max Iters", min=10, max=1000, step=10, initial_value=500
    )

gui_stop: dict[str, viser.GuiButtonHandle] = {}
gui_costs: dict[str, viser.GuiInputHandle] = {}
for name, (r, g, b) in SOLVER_COLORS.items():
    with server.gui.add_folder(name):
        gui_stop[name] = server.gui.add_button("Stop")
        gui_costs[name] = server.gui.add_number("Cost", 0.0, disabled=True)

with server.gui.add_folder("Playback"):
    gui_play = server.gui.add_button("Play / Pause")
    gui_speed = server.gui.add_slider(
        "Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0
    )
    gui_timestep = server.gui.add_slider(
        "Timestep", min=0, max=T - 1, step=1, initial_value=0
    )

gui_iter = server.gui.add_number("Iteration", 0, disabled=True)


# --- State ---------------------------------------------------------------- #


@dataclass
class SolverState:
    """Mutable optimization state for a single solver."""

    q: jnp.ndarray
    cost_history: list[float] = field(default_factory=list)
    iteration: int = 0
    converged: bool = False
    stopped: bool = False


def make_fresh_q() -> jnp.ndarray:
    """Flat initial config: tile home pose across all T timesteps."""
    return jnp.tile(PANDA_HOME, (T, 1)).reshape(-1)


COST_TOL = 1e-6
FPS = 20
TRAIL_UPDATE_INTERVAL = 1

states: dict[str, SolverState] = {
    name: SolverState(q=make_fresh_q()) for name in SOLVER_COLORS
}
trail_handles: dict[str, object] = {}
optimizing = True
paused = [False]
step_once = [False]
playing = [False]
playback_accum = [0.0]
last_time = [time.time()]
reset_signal = [False]


# --- GUI callbacks -------------------------------------------------------- #


@gui_reset.on_click
def _(_: viser.GuiEvent) -> None:
    """Reset all solvers and return to optimization."""
    reset_signal[0] = True
    global optimizing
    optimizing = True
    playing[0] = False


@gui_pause.on_click
def _(_: viser.GuiEvent) -> None:
    """Toggle pause during optimization."""
    paused[0] = not paused[0]


@gui_step.on_click
def _(_: viser.GuiEvent) -> None:
    """Advance all active solvers by one iteration."""
    step_once[0] = True


@gui_play.on_click
def _(_: viser.GuiEvent) -> None:
    """Switch to playback with whatever state each solver has reached."""
    playing[0] = not playing[0]
    last_time[0] = time.time()
    playback_accum[0] = 0.0
    global optimizing
    optimizing = False


for _name in SOLVER_COLORS:

    def _make_cb(n: str):
        def cb(_: viser.GuiEvent) -> None:
            states[n].stopped = True
            states[n].converged = True
            if states[n].cost_history:
                print(
                    f"[Stopped] {n} at iter {states[n].iteration}, cost={states[n].cost_history[-1]:.6e}"
                )
            else:
                print(f"[Stopped] {n}")

        return cb

    gui_stop[_name].on_click(_make_cb(_name))


# --- Visualization helpers ------------------------------------------------ #


def get_ee_positions(q_flat: jnp.ndarray) -> onp.ndarray:
    """Batch FK: flat (T*n_joints,) -> (T, 3) end-effector positions."""
    qu = q_flat.reshape(T, N_JOINTS)
    fk = jax.vmap(robot.forward_kinematics)(qu)
    return onp.asarray(jaxlie.SE3(fk[:, target_link_index]).translation())


def draw_trail(name: str, positions: onp.ndarray, color: tuple[int, int, int]) -> None:
    """Draw (or redraw) a solver's end-effector trail as colored line segments."""
    if name in trail_handles:
        trail_handles[name].remove()
    if positions.shape[0] < 2:
        return
    trail_handles[name] = server.scene.add_line_segments(
        f"/trails/{name}",
        points=onp.stack([positions[:-1], positions[1:]], axis=1),
        colors=color,
        line_width=3.0,
    )


# --- Main loop ------------------------------------------------------------ #

while True:
    # Handle reset.
    if reset_signal[0]:
        reset_signal[0] = False
        states = {name: SolverState(q=make_fresh_q()) for name in SOLVER_COLORS}
        for h in trail_handles.values():
            h.remove()
        trail_handles.clear()
        optimizing = True
        paused[0] = False
        playing[0] = False
        gui_timestep.value = 0
        for name in SOLVER_COLORS:
            solver_robots[name].update_cfg(onp.asarray(PANDA_HOME))
            gui_costs[name].value = 0.0
        gui_iter.value = 0

    # Target poses: track position from target_traj, hold orientation from home FK.
    ee_target = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(jnp.tile(fk_ee.rotation().wxyz, (T, 1))), target_traj
    )

    if optimizing:
        should_step = (not paused[0]) or step_once[0]
        step_once[0] = False

        if should_step:
            all_converged = True
            for name, color in SOLVER_COLORS.items():
                # Bail if the user hit play or reset mid-iteration.
                if not optimizing or reset_signal[0]:
                    break

                st = states[name]
                if st.converged:
                    continue
                all_converged = False

                # Compute direction + line search.
                direction = DIRECTION_FNS[name](st.q, ee_target)
                st.q = line_search(st.q, direction, ee_target)

                # Track cost.
                cost = float(loss(st.q, ee_target))
                st.cost_history.append(cost)
                st.iteration += 1

                # Convergence check.
                if len(st.cost_history) >= 2:
                    if abs(st.cost_history[-1] - st.cost_history[-2]) < COST_TOL:
                        st.converged = True
                if st.iteration >= int(gui_max_iters.value):
                    st.converged = True

                # Update viz.
                solver_robots[name].update_cfg(
                    onp.asarray(st.q.reshape(T, N_JOINTS)[0])
                )
                gui_costs[name].value = cost
                if st.iteration % TRAIL_UPDATE_INTERVAL == 0 or st.converged:
                    draw_trail(name, get_ee_positions(st.q), color)

            gui_iter.value = max(st.iteration for st in states.values())

            if all_converged and optimizing:
                optimizing = False
                print("\n=== All converged/stopped ===")
                for name, st in states.items():
                    tag = "stopped" if st.stopped else "converged"
                    cost_str = (
                        f"{st.cost_history[-1]:.6e}" if st.cost_history else "N/A"
                    )
                    print(
                        f"  {name:20s}: {st.iteration} iters, cost={cost_str} ({tag})"
                    )
                for name, color in SOLVER_COLORS.items():
                    draw_trail(name, get_ee_positions(states[name].q), color)

    else:
        # Playback: animate all three robots along their solved trajectories.
        if playing[0]:
            now = time.time()
            dt = now - last_time[0]
            last_time[0] = now
            playback_accum[0] += dt * gui_speed.value * FPS
            while playback_accum[0] >= 1.0:
                playback_accum[0] -= 1.0
                gui_timestep.value = (int(gui_timestep.value) + 1) % T

        t_idx = int(gui_timestep.value)
        for name in SOLVER_COLORS:
            q_traj = states[name].q.reshape(T, N_JOINTS)
            solver_robots[name].update_cfg(onp.asarray(q_traj[t_idx]))

    time.sleep(1.0 / FPS)
