"""
Time integration schemes for second-order ODE systems of the form

    M @ q_ddot + C @ q_dot + K @ q = Q
"""

from __future__ import annotations

import numpy as np


class NewmarkIntegrator:
    """
    Newmark-beta integrator with iterative residual correction.
    """

    def __init__(
        self,
        gamma: float = 0.51,
        beta: float = 0.25,
        eps: float = 1e-6,
        max_iter: int = 100,
    ) -> None:
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.max_iter = max_iter
        self.last_iteration_count = 0

    def step(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ddot: np.ndarray,
        M: np.ndarray,
        K: np.ndarray,
        C: np.ndarray,
        Q_new: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Predictor
        q_new = q + dt * q_dot + 0.5 * dt**2 * q_ddot
        q_dot_new = q_dot + dt * q_ddot
        q_ddot_new = q_ddot.copy()

        coef1 = self.gamma / (self.beta * dt)
        coef2 = 1.0 / (self.beta * dt**2)
        K_eff = K + coef1 * C + coef2 * M

        for it in range(1, self.max_iter + 1):
            resid = Q_new - M @ q_ddot_new - C @ q_dot_new - K @ q_new
            if np.max(np.abs(resid)) < self.eps:
                self.last_iteration_count = it - 1
                return q_new, q_dot_new, q_ddot_new

            delta = np.linalg.solve(K_eff, resid)
            q_new = q_new + delta
            q_dot_new = q_dot_new + coef1 * delta
            q_ddot_new = q_ddot_new + coef2 * delta

        self.last_iteration_count = self.max_iter
        print(f"NewmarkIntegrator: did not converge in {self.max_iter} iterations")
        return q_new, q_dot_new, q_ddot_new
