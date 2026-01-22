import jax
import jax.numpy as jnp

from solution import least_squares_loss, get_problem_data, lstsq_params_closed_form


def test_problem_data_shapes() -> None:
    current = jnp.array([0.0, 1.0, 2.0])
    tau = jnp.array([1.0, 2.0, 3.0])
    A, y = get_problem_data(current, tau)
    assert A.shape == (3, 3)
    assert y.shape == (3,)


def test_least_squares_loss_shape() -> None:
    current = jnp.array([0.0, 1.0, 2.0])
    tau = jnp.array([1.0, 2.0, 3.0])
    A, y = get_problem_data(current, tau)
    x = jnp.array([0.0, 0.0, 0.0])
    loss = least_squares_loss(x, A, y)
    assert loss.shape == ()


def test_lstsq_params_closed_form_shape() -> None:
    current = jnp.array([0.0, 1.0, 2.0])
    tau = jnp.array([1.0, 2.0, 3.0])
    A, y = get_problem_data(current, tau)
    x_opt = lstsq_params_closed_form(A, y)
    assert x_opt.shape == (3,)


def test_least_squares_loss_jittable() -> None:
    current = jnp.array([0.0, 1.0, 2.0])
    tau = jnp.array([1.0, 2.0, 3.0])
    A, y = get_problem_data(current, tau)
    x = jnp.array([0.0, 0.0, 0.0])
    jitted_loss = jax.jit(least_squares_loss)
    loss = jitted_loss(x, A, y)
    assert loss.shape == ()


def test_least_squares_loss_grad_jittable() -> None:
    current = jnp.array([0.0, 1.0, 2.0])
    tau = jnp.array([1.0, 2.0, 3.0])
    A, y = get_problem_data(current, tau)
    x = jnp.array([0.0, 0.0, 0.0])
    jitted_grad = jax.jit(jax.grad(least_squares_loss))
    grad = jitted_grad(x, A, y)
    assert grad.shape == (3,)
