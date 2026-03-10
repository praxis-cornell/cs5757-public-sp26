import unittest

import cvxpy as cvx
import jax
from jax import numpy as jnp

from solution import (
    affinize_dynamics,
    affinize_constraint,
    linearize,
    ilqr_backward_step,
    signed_distance,
    OBSTACLE_CENTERS,
)


class TestRetargeting(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        self.m = 10
        self.n = 4
        self.T = 15
        self.x = jnp.zeros(self.n)
        self.u = jnp.zeros(self.m)

    def f_trivial(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """A trivial dynamics function for testing."""
        return x

    def cost_trivial(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """A trivial cost function for testing."""
        return jnp.sum(x**2) + jnp.sum(u**2)

    def trivial_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        """A trivial constraint function for testing."""
        return x

    def test_linearize_shapes(self):

        Q, R, S, q, r, A, B = linearize(
            self.f_trivial, self.cost_trivial, self.x, self.u
        )

        assert Q.shape == (
            self.n,
            self.n,
        ), f"Expected Q shape {(self.n, self.n)}, got {Q.shape}"
        assert R.shape == (
            self.m,
            self.m,
        ), f"Expected R shape {(self.m, self.m)}, got {R.shape}"
        assert S.shape == (
            self.n,
            self.m,
        ), f"Expected S shape {(self.n, self.m)}, got {S.shape}"
        assert q.shape == (self.n,), f"Expected q shape {(self.n,)}, got {q.shape}"
        assert r.shape == (self.m,), f"Expected r shape {(self.m,)}, got {r.shape}"
        assert A.shape == (
            self.n,
            self.n,
        ), f"Expected A shape {(self.n, self.n)}, got {A.shape}"
        assert B.shape == (
            self.n,
            self.m,
        ), f"Expected B shape {(self.n, self.m)}, got {B.shape}"

    def test_ilqr_backward_step_shapes(self):
        Q, R, S, q, r, A, B = linearize(
            self.f_trivial, self.cost_trivial, self.x, self.u
        )

        P = jnp.zeros((self.n, self.n))
        p = jnp.zeros(self.n)

        P_new, p_new, K, k = ilqr_backward_step(P, p, Q, R, S, q, r, A, B)

        assert P_new.shape == (
            self.n,
            self.n,
        ), f"Expected P_new shape {(self.n, self.n)}, got {P_new.shape}"
        assert p_new.shape == (
            self.n,
        ), f"Expected p_new shape {(self.n,)}, got {p_new.shape}"
        assert K.shape == (
            self.m,
            self.n,
        ), f"Expected K shape {(self.m, self.n)}, got {K.shape}"
        assert k.shape == (self.m,), f"Expected k shape {(self.m,)}, got {k.shape}"

    def test_sdf_shape(self):
        state = jnp.zeros(6)
        sdf_values = signed_distance(state)

        assert sdf_values.shape == (
            OBSTACLE_CENTERS.shape[0],
        ), f"Expected sdf_values shape {(OBSTACLE_CENTERS.shape[0],)}, got {sdf_values.shape}"

    def test_affinize_shape(self):
        A, B, c = affinize_dynamics(self.f_trivial, self.x, self.u)
        assert A.shape == (
            self.n,
            self.n,
        ), f"Expected A shape {(self.n, self.n)}, got {A.shape}"
        assert B.shape == (
            self.n,
            self.m,
        ), f"Expected B shape {(self .n, self.m)}, got {B.shape}"
        assert c.shape == (self.n,), f"Expected c shape {(self.n,)}, got {c.shape}"

    def test_affinize_constraints_shape(self):
        G_con, h_con = affinize_constraint(self.trivial_constraint, self.x)
        assert G_con.shape == (
            self.n,
            self.n,
        ), f"Expected G_con shape {(self.n, self.n)}, got {G_con.shape}"
        assert h_con.shape == (
            self.n,
        ), f"Expected h_con shape {(self.n,)}, got {h_con.shape}"


if __name__ == "__main__":
    unittest.main()
