import jax.numpy as jnp


def get_problem_data(
    I: jnp.ndarray, tau: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generates the least-squares problem data."""
    A = jnp.stack([jnp.ones_like(I), I, I**2], axis=-1)

    return A, tau


def least_squares_loss(x: jnp.ndarray, A: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Computes the least-squares loss."""
    residuals = A @ x - y
    return jnp.mean(jnp.square(residuals))


def lstsq_params_closed_form(A: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Computes the least-squares solution using the closed-form formula."""
    return jnp.linalg.inv(A.T @ A) @ A.T @ y
