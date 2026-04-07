import unittest

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

from solution import (
    query_zoh,
    shift_warmstart,
    mppi_update,
    cem_update,
    ps_update,
    SlamState,
    state_add,
    residual_odometry,
    residual_landmark,
    residual_lm_prior,
    residual_prior,
    build_stacked_residual_fn,
    robustify_huber,
    gauss_newton_step,
)


class TestSamplingMPC(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        self.K = 64  # number of samples
        self.n = 8  # number of knots
        self.d_u = 6  # control dimension
        self.H = 20  # horizon (query points)

    def test_query_zoh_shape(self):
        knot_times = jnp.arange(self.n, dtype=float)
        knot_values = jnp.zeros((self.n, self.d_u))
        query_times = jnp.linspace(0, self.n - 1, self.H)

        result = query_zoh(knot_times, knot_values, query_times)

        assert result.shape == (
            self.H,
            self.d_u,
        ), f"Expected shape {(self.H, self.d_u)}, got {result.shape}"

    def test_shift_warmstart_shape(self):
        knot_times = np.arange(self.n, dtype=float)
        knot_values = np.zeros((self.n, self.d_u))

        result = shift_warmstart(knot_times, knot_values, ctrl_steps=2)

        assert result.shape == (
            self.n,
            self.d_u,
        ), f"Expected shape {(self.n, self.d_u)}, got {result.shape}"

    def test_mppi_update_shape(self):
        z_samples = jnp.zeros((self.K, self.n, self.d_u))
        costs = jnp.ones(self.K)

        result = mppi_update(z_samples, costs)

        assert result.shape == (
            self.n,
            self.d_u,
        ), f"Expected shape {(self.n, self.d_u)}, got {result.shape}"

    def test_cem_update_shape(self):
        z_samples = jnp.zeros((self.K, self.n, self.d_u))
        costs = jnp.ones(self.K)

        mu, sigma = cem_update(z_samples, costs)

        assert mu.shape == (
            self.n,
            self.d_u,
        ), f"Expected mu shape {(self.n, self.d_u)}, got {mu.shape}"
        assert sigma.shape == (
            self.n,
            self.d_u,
        ), f"Expected sigma shape {(self.n, self.d_u)}, got {sigma.shape}"

    def test_ps_update_shape(self):
        z_samples = jnp.zeros((self.K, self.n, self.d_u))
        costs = jnp.ones(self.K)

        result = ps_update(z_samples, costs)

        assert result.shape == (
            self.n,
            self.d_u,
        ), f"Expected shape {(self.n, self.d_u)}, got {result.shape}"


class TestSLAM(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(0)
        self.T = 5  # number of poses
        self.N = 3  # number of landmarks

        self.drone_state = jaxlie.SE3.identity(batch_axes=(self.T,))
        self.landmark_pos = jnp.zeros((self.N, 3))
        self.state = SlamState(self.drone_state, self.landmark_pos)

    def test_state_add_shape(self):
        n_vars = self.T * 6 + self.N * 3
        delta = jnp.zeros(n_vars)

        new_state = state_add(self.state, delta)

        assert new_state.drone_state.wxyz_xyz.shape == (
            self.T,
            7,
        ), f"Expected drone shape {(self.T, 7)}, got {new_state.drone_state.wxyz_xyz.shape}"
        assert new_state.landmark_pos.shape == (
            self.N,
            3,
        ), f"Expected landmark shape {(self.N, 3)}, got {new_state.landmark_pos.shape}"

    def test_residual_odometry_shape(self):
        T_i = jaxlie.SE3.identity()
        T_j = jaxlie.SE3.identity()
        T_ij = jaxlie.SE3.identity()
        sqrt_info = jnp.eye(6)

        r = residual_odometry(T_i, T_j, T_ij, sqrt_info)

        assert r.shape == (6,), f"Expected shape (6,), got {r.shape}"

    def test_residual_landmark_shape(self):
        T_i = jaxlie.SE3.identity()
        p_l = jnp.zeros(3)
        z_meas = jnp.zeros(3)
        sqrt_info = jnp.eye(3)

        r = residual_landmark(T_i, p_l, z_meas, sqrt_info)

        assert r.shape == (3,), f"Expected shape (3,), got {r.shape}"

    def test_residual_lm_prior_shape(self):
        p_l = jnp.zeros(3)
        p_prior = jnp.zeros(3)
        sqrt_info = jnp.eye(3)

        r = residual_lm_prior(p_l, p_prior, sqrt_info)

        assert r.shape == (3,), f"Expected shape (3,), got {r.shape}"

    def test_residual_prior_shape(self):
        T_0 = jaxlie.SE3.identity()
        T_prior = jaxlie.SE3.identity()

        r = residual_prior(T_0, T_prior)

        assert r.shape == (6,), f"Expected shape (6,), got {r.shape}"

    def test_robustify_huber_shape(self):
        r = jnp.ones(6)

        result = robustify_huber(r, delta=1.0)

        assert result.shape == (6,), f"Expected shape (6,), got {result.shape}"

    def test_gauss_newton_step_returns(self):
        # Minimal factor graph: 1 pose, 1 landmark, prior only
        state = SlamState(
            jaxlie.SE3.identity(batch_axes=(1,)),
            jnp.zeros((1, 3)),
        )

        def dummy_residual(s, d):
            s2 = state_add(s, d)
            return residual_prior(
                jaxlie.SE3(wxyz_xyz=s2.drone_state.wxyz_xyz[0]),
                jaxlie.SE3.identity(),
            )

        new_state, cost = gauss_newton_step(state, dummy_residual)

        assert new_state.drone_state.wxyz_xyz.shape == (1, 7)
        assert cost.shape == ()


if __name__ == "__main__":
    unittest.main()
