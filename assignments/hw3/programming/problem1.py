"""
Problem 3.1 — iLQR and feedback under noise (acrobot)

Solves a swing-up with iLQR, then compares open-loop vs. closed-loop
execution under increasing levels of additive process noise.

CS 5757, Cornell University
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

from solution import (
    acrobot,
    discretize,
    rollout,
    ilqr,
    simulate,
    quadratic_cost,
    quadratic_terminal_cost,
)

jax.config.update("jax_enable_x64", True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# Problem setup
n = 4
dt = 0.01
N = 300

Q = jnp.diag(jnp.array([0.1, 0.01, 0.01, 0.01]))
R = jnp.diag(jnp.array([0.01]))
Q_T = jnp.diag(jnp.array([100.0, 100.0, 1.0, 1.0]))

cost = lambda s, a: quadratic_cost(s, a, Q, R)
cost_term = lambda s: quadratic_terminal_cost(s, Q_T)

fd = jax.jit(discretize(acrobot, dt))
s0 = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

# ── Solve with iLQR ──

a_init = jnp.zeros((N, 1))
s_init = rollout(fd, s0, a_init)

print("Computing iLQR solution ... ", end="", flush=True)
start = time.time()
s_nom, a_nom, K_seq, k_seq = ilqr(fd, cost, cost_term, s0, s_init, a_init)
print(f"done! ({time.time() - start:.2f} s)")

# ── Compare open-loop vs. feedback across noise levels ──

print("Simulating under noise ...")
key = random.PRNGKey(0)
noise_levels = [0.0, 0.01, 0.05, 0.1, 0.5, 0.75]
labels = (r"$\theta_1$", r"$\theta_2$", r"$\dot{\theta}_1$", r"$\dot{\theta}_2$")

fig, axes = plt.subplots(
    len(noise_levels), n, figsize=(16, 1.5 * len(noise_levels)), sharex=True
)
for i, sigma in enumerate(noise_levels):
    key, k1, k2 = random.split(key, 3)
    s_ol, _ = simulate(
        fd, s0, s_nom, a_nom, K_seq, k_seq, sigma, k1, use_feedback=False
    )
    s_fb, a_fb = simulate(
        fd, s0, s_nom, a_nom, K_seq, k_seq, sigma, k2, use_feedback=True
    )

    J_ol = jax.vmap(cost)(s_ol[:-1], a_nom).sum() + cost_term(s_ol[-1])
    J_fb = jax.vmap(cost)(s_fb[:-1], a_fb).sum() + cost_term(s_fb[-1])

    for j in range(n):
        ax = axes[i, j]
        ax.plot(
            np.arange(N + 1) * dt,
            np.array(s_nom[:, j]),
            "k--",
            alpha=0.5,
            label="Nominal",
        )
        ax.plot(
            np.arange(N + 1) * dt,
            np.array(s_ol[:, j]),
            "r",
            alpha=0.7,
            label="Open-loop",
        )
        ax.plot(
            np.arange(N + 1) * dt,
            np.array(s_fb[:, j]),
            "b",
            alpha=0.7,
            label="Feedback",
        )
        if i == 0:
            ax.set_title(labels[j])
        if j == 0:
            ax.set_ylabel(rf"$\sigma = {sigma}$")
        if i == len(noise_levels) - 1:
            ax.set_xlabel("Time [s]")
        if i == 0 and j == 0:
            ax.legend(fontsize=8)

    print(f"  σ={sigma:.2f}:  OL cost = {J_ol:.2f},  FB cost = {J_fb:.2f}")

fig.suptitle("Open-loop vs. Feedback under Process Noise", fontsize=14)
plt.tight_layout()
plt.savefig("problem1_noise_comparison.pdf", bbox_inches="tight")
plt.show()
