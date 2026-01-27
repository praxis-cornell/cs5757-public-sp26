import time
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import plotly.graph_objects as go
import viser

from jax import Array
from judo.visualizers.model import ViserMjModel
from mujoco import mjx

# --- Configuration & Constants ---
XML = """
<mujoco>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="wall">
      <geom name="wall" type="box" size="0.1 1.0 0.5" pos="2 0 0.5" rgba="0.8 0.2 0.2 1"/>
    </body>
    <body name="ball" pos="0 0 1">
      <joint type="free"/>
      <geom size="0.1" type="sphere" rgba="0.2 0.2 0.8 1"/>
    </body>
    <geom name="floor" type="plane" size="0 0 0.1" rgba="0.8 0.9 0.8 1"/>
  </worldbody>
</mujoco>
"""


class PhysicsEngine:
    """Simple physics engine for simulating a ball launched at a wall.

    Args:
        xml: MuJoCo XML string defining the simulation.
        n_steps: Number of simulation steps to run per rollout.
        v_mag: Magnitude of the ball's launch velocity.
    """

    def __init__(self, xml: str, n_steps: int = 200, v_mag: float = 5.0):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.mjx_model = mjx.put_model(self.model)
        self.n_steps = n_steps
        self.v_mag = v_mag

        # JIT compile the rollout
        self._rollout_jit = jax.jit(self._run_sim)
        self._grad_jit = jax.jit(jax.jacfwd(lambda t: self._run_sim(t)[-1][0]))

    def _run_sim(self, theta: float) -> Array:
        """Runs a simulation rollout with the ball launched at angle theta.
        Args:
            theta: Launch angle in radians.
        Returns:
            Trajectory of ball positions over time.
        """
        d = mjx.make_data(self.model)
        vx, vz = self.v_mag * jnp.cos(theta), self.v_mag * jnp.sin(theta)
        d = d.replace(qvel=d.qvel.at[0].set(vx).at[2].set(vz))

        def step(d: mjx.Data, _) -> tuple[mjx.Data, Array]:
            d_next = mjx.step(self.mjx_model, d)
            return d_next, d_next.qpos

        _, traj = jax.lax.scan(step, d, None, length=self.n_steps)
        return traj

    def get_trajectory(self, theta: float) -> np.ndarray:
        """Returns the trajectory of the ball for a given launch angle theta."""
        return np.array(self._rollout_jit(theta))

    def get_metrics(self, theta: float) -> tuple[float, float]:
        """Returns the final x position and its gradient w.r.t. theta."""
        traj = self._rollout_jit(theta)
        final_x = traj[-1, 0]
        grad = self._grad_jit(theta)
        return float(final_x), float(grad)


def get_kinematic_arc(theta: float, v0: float = 5.0, steps: int = 30) -> np.ndarray:
    """Computes the kinematic arc of the ball launched at angle theta."""
    t = np.linspace(0, 0.8, steps)
    x = v0 * np.cos(theta) * t
    z = 1.0 + (v0 * np.sin(theta) * t) - (0.5 * 9.81 * t**2)
    return np.stack([x, np.zeros_like(x), z], axis=1)


def build_plot(
    x_range: np.ndarray,
    y_range: np.ndarray,
    current_x: float,
    current_y: float,
    title: str,
    label: str,
) -> go.Figure:
    """Builds a Plotly figure for the optimization landscape."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_range, y=y_range, name="Landscape"))
    fig.add_trace(
        go.Scatter(x=[current_x], y=[current_y], marker=dict(color="red", size=10))
    )
    fig.update_layout(
        title=title,
        yaxis_title=label,
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig


def main():
    """Main application loop."""
    server = viser.ViserServer()
    engine = PhysicsEngine(XML)
    viser_model = ViserMjModel(server, mujoco.MjSpec.from_string(XML))

    # Cache optimization landscape
    thetas = np.linspace(0, np.pi / 2, 100)
    landscape_x = [engine.get_metrics(t)[0] for t in thetas]
    landscape_g = [engine.get_metrics(t)[1] for t in thetas]

    # UI Components
    with server.gui.add_folder("Sim Controls"):
        gui_theta = server.gui.add_slider("Launch Angle", 0.0, 1.57, 0.01, 0.78)
        gui_playing = server.gui.add_checkbox("Active Playback", True)

    with server.gui.add_folder("Analysis"):
        plot_x = server.gui.add_plotly(go.Figure())
        plot_g = server.gui.add_plotly(go.Figure())

    # App State
    state = {"traj": engine.get_trajectory(0.78), "frame": 0}
    mj_data = mujoco.MjData(engine.model)

    @gui_theta.on_update
    def _(_):
        theta = gui_theta.value
        state["traj"] = engine.get_trajectory(theta)
        cur_x, cur_g = engine.get_metrics(theta)

        # Update Plots
        plot_x.figure = build_plot(
            thetas, landscape_x, theta, cur_x, "Final X Pos", "Meters"
        )
        plot_g.figure = build_plot(
            thetas, landscape_g, theta, cur_g, "Gradient", "dX/dTheta"
        )

        # Update Ghost Arc
        server.scene.add_spline_catmull_rom(
            "/prediction", get_kinematic_arc(theta), color=(255, 255, 0), line_width=2.0
        )

    # Initial trigger
    # Define the update logic in a named function
    def update_scene(_=None):
        theta = gui_theta.value
        state["traj"] = engine.get_trajectory(theta)
        cur_x, cur_g = engine.get_metrics(theta)

        plot_x.figure = build_plot(
            thetas, landscape_x, theta, cur_x, "Final X Pos", "Meters"
        )
        plot_g.figure = build_plot(
            thetas, landscape_g, theta, cur_g, "Gradient", "dX/dTheta"
        )

        server.scene.add_spline_catmull_rom(
            "/prediction", get_kinematic_arc(theta), color=(255, 255, 0), line_width=2.0
        )

    # Attach it to the slider
    gui_theta.on_update(update_scene)

    # Manually call it once to initialize the UI/Plots
    update_scene()

    while True:
        if gui_playing.value:
            state["frame"] = (state["frame"] + 1) % len(state["traj"])
            mj_data.qpos[:] = state["traj"][state["frame"]]
            mujoco.mj_forward(engine.model, mj_data)
            viser_model.set_data(mj_data)

        time.sleep(engine.model.opt.timestep)


if __name__ == "__main__":
    main()
