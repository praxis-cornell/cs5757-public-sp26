import jax
import jax.numpy as jnp

from solution import (
    build_constraints,
    solve_spline_1d,
    evaluate_spline_1d,
    DroneState,
    f_drone,
    rk4_step,
    compute_attitude_control,
)


def test_build_constraints_shapes() -> None:
    """Checks the return type and shape of build_constraints."""
    # Create dummy data.
    x0 = 0.0
    gates = jnp.zeros(2)
    dt = jnp.ones(2)

    # Call function.
    A, b = build_constraints(x0, gates, dt)

    # Check types and shapes.
    assert isinstance(A, jnp.ndarray), "Returned A is not a jnp.ndarray"
    assert isinstance(b, jnp.ndarray), "Returned b is not a jnp.ndarray"
    assert A.shape == (7, 8)
    assert b.shape == (7,)


def test_solve_spline_1d_shape() -> None:
    """Checks the return type and shape of solve_spline_1d."""
    # Create dummy data
    x0 = 0.0
    gates = jnp.zeros(2)
    dt = jnp.ones(2)

    # Call function
    coeffs = solve_spline_1d(x0, gates, dt)

    # Check return type and shape
    assert isinstance(coeffs, jnp.ndarray)
    assert coeffs.shape == (8,)


def test_solve_spline_vmappable() -> None:
    """Checks that solve_spline_1d is vmappable over space."""
    x0 = jnp.zeros(3)
    gates = jnp.ones((3, 2))
    dt = jnp.ones(2)

    coeffs = jax.vmap(solve_spline_1d, in_axes=(0, 0, None))(x0, gates, dt)

    assert coeffs.shape == (3, 8)


def test_evaluate_spline_1d_shape() -> None:
    """Checks the return type and shape of evaluate_spline_1d."""
    # Create dummy data
    coeffs = jnp.arange(8.0)
    t = 0.5
    dt = jnp.ones(2)

    # Call function
    position, velocity, acceleration = evaluate_spline_1d(coeffs, t, dt)

    # Check return type and shape
    assert isinstance(position, jnp.ndarray)
    assert position.shape == ()
    assert isinstance(velocity, jnp.ndarray)
    assert velocity.shape == ()
    assert isinstance(acceleration, jnp.ndarray)
    assert acceleration.shape == ()


def test_evaluate_spline_vmappable() -> None:
    """Checks that evaluate_spline_1d is vmappable over time and space."""
    # Create dummy data
    coeffs = jnp.ones((3, 8))
    ts = jnp.linspace(0.0, 2.0, num=8)
    dt = jnp.ones(2)

    # Vectorize function over time
    def eval_at_t(t: float) -> jnp.ndarray:
        return jax.vmap(evaluate_spline_1d, in_axes=(0, None, None))(coeffs, t, dt)

    positions, velocities, accelerations = jax.vmap(eval_at_t)(ts)

    # Check return type and shape
    assert positions.shape == (8, 3)
    assert velocities.shape == (8, 3)
    assert accelerations.shape == (8, 3)


def test_f_drone_shape() -> None:
    """Checks the return type and shape of f_drone."""
    # Create dummy data
    state = DroneState(
        posW=jnp.zeros(3),
        R=jnp.eye(3),
        velW=jnp.zeros(3),
        omegaB=jnp.zeros(3),
    )
    control = jnp.zeros(4)

    # Call function
    dposW, omegaB, dvelW, domegaB = f_drone(state, control)

    # Check return type and shape
    assert isinstance(dposW, jnp.ndarray)
    assert dposW.shape == (3,)
    assert isinstance(omegaB, jnp.ndarray)
    assert omegaB.shape == (3,)
    assert isinstance(dvelW, jnp.ndarray)
    assert dvelW.shape == (3,)
    assert isinstance(domegaB, jnp.ndarray)
    assert domegaB.shape == (3,)


def test_rk4_step_shape() -> None:
    """Checks the return type and shape of rk4_step."""
    # Create dummy data
    state = DroneState(
        posW=jnp.zeros(3),
        R=jnp.eye(3),
        velW=jnp.zeros(3),
        omegaB=jnp.zeros(3),
    )
    control = jnp.zeros(4)
    dt = 0.1

    # Call function
    next_state = rk4_step(state, control, dt)

    # Check return type and shape
    assert isinstance(next_state, DroneState)
    assert next_state.posW.shape == (3,)
    assert next_state.R.shape == (3, 3)
    assert next_state.velW.shape == (3,)
    assert next_state.omegaB.shape == (3,)


def test_compute_attitude_control_shape() -> None:
    """Checks the return type and shape of compute_attitude_control."""
    # Create dummy data
    R = jnp.eye(3)
    R_d = jnp.eye(3)
    omegaB = jnp.zeros(3)

    # Call function
    tau = compute_attitude_control(R, R_d, omegaB)

    # Check return type and shape
    assert isinstance(tau, jnp.ndarray)
    assert tau.shape == (3,)
