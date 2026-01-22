import jax
import jax.numpy as jnp


def get_problem_data(
    I: jnp.ndarray, tau: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generates the least-squares problem data."""
    raise NotImplementedError("Not implemented")


def least_squares_loss(x: jnp.ndarray, A: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Computes the least-squares loss."""
    raise NotImplementedError("Not implemented")

def lstsq_params_closed_form(A: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Computes the least-squares solution using the closed-form formula."""
    raise NotImplementedError("Not implemented")