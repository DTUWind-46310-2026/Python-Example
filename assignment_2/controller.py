from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import Simulation

import numpy as np


class ControllerBase(ABC):

    @abstractmethod
    def step(self, simulation):
        pass

    def simulation_init(self, simulation: Simulation):
        pass

    @property
    @abstractmethod
    def generator_torque(self) -> float:
        pass

    @property
    @abstractmethod
    def setpoint_pitch(self) -> np.ndarray:
        pass


class PIController(ControllerBase):
    def __init__(self, Kp=1.5, Ki=0.64, KK=14, omega_ref_factor=1.0, above_rated_mode="power") -> None:
        self.K_omega_opt = 0
        self.Kp = Kp
        self.Ki = Ki
        self.KK = np.deg2rad(KK)
        self.above_rated_mode = above_rated_mode
        self._gen_torque = 0
        self._gen_torque_rated = 0
        self._sp_pitch_i = 0
        self._setpoint_pitch = 0
        self._blades = np.asarray([])
        self.omega_ref_factor = omega_ref_factor
        self.omega_ref = 0

        self._torque_control = 0
        self._pitch_control = 0

    def simulation_init(self, simulation: Simulation):
        self.K_omega_opt = (
            0.5
            * simulation.aerodynamics.rho
            * (simulation.structure.R / simulation.aerodynamics.tsr_opt) ** 3
            * (np.pi * simulation.structure.R**2)
            * simulation.aerodynamics.CP_max
        )
        self._sp_pitch_i = simulation.structure.pitch[0]
        self._setpoint_pitch = simulation.structure.pitch[0]
        self._blades = np.asarray([float(i) for i in range(simulation.structure.n_blades)])
        self.omega_ref = simulation.aerodynamics.omega_rated * self.omega_ref_factor
        self._gen_torque_rated = simulation.aerodynamics.P_rated / simulation.aerodynamics.omega_rated

    def step(self, simulation: Simulation):
        self.step_generator_torque(simulation)
        self.step_collective_pitch(simulation)

    def step_generator_torque(self, simulation: Simulation):
        if simulation.structure.omega_shaft <= self.omega_ref:
            self._gen_torque = self.K_omega_opt * simulation.structure.omega_shaft**2
            self._torque_control = 0
        else:
            if self.above_rated_mode == "power":
                self._gen_torque = simulation.aerodynamics.P_rated / simulation.structure.omega_shaft
                self._torque_control = 1
            elif self.above_rated_mode == "torque":
                self._gen_torque = simulation.aerodynamics.P_rated / simulation.aerodynamics.omega_rated
                self._torque_control = 2
            else:
                raise NotImplementedError(f"{self.above_rated_mode=} when implemented are 'power', 'torque'.")

    def step_collective_pitch(self, simulation: Simulation):
        omega_diff = simulation.structure.omega_shaft - self.omega_ref
        GK = 1 / (1 + simulation.structure.pitch[0] / self.KK)
        sp_pitch_p = GK * self.Kp * omega_diff
        self._sp_pitch_i += GK * self.Ki * omega_diff * simulation.dt
        self._sp_pitch_i = np.clip(self._sp_pitch_i, *simulation.structure.pitch_range)
        self._setpoint_pitch = np.clip(self._sp_pitch_i + sp_pitch_p, *simulation.structure.pitch_range)

        if self._setpoint_pitch > simulation.structure.pitch_range[0]:
            self._pitch_control = 1
        else:
            self._pitch_control = 0

    @property
    def generator_torque(self):
        return self._gen_torque

    @property
    def setpoint_pitch(self) -> np.ndarray:
        return np.full_like(self._blades, self._setpoint_pitch)

    @property
    def modes(self) -> tuple[int, int]:
        """
        Torque control modes:
            0: k-omega²
            1: constant power
            2: constant torque

        Pitch control modes:
            0: deactivatd
            1: activated

        Returns
        -------
        tuple[int, int]
            Torque control mode, Pitch control mode
        """
        return self._torque_control, self._pitch_control

    def power(self, simulation: Simulation):
        return self.generator_torque * simulation.structure.omega_shaft


class ConstantRotSpeedController(ControllerBase):
    def __init__(self) -> None:
        self._gen_torque = 0
        self._pitch = np.asarray([])

    def simulation_init(self, simulation: Simulation):
        self._pitch = simulation.structure.pitch

    def step(self, simulation: Simulation):
        self._gen_torque = simulation.aerodynamics.torque

    @property
    def generator_torque(self) -> float:
        return self._gen_torque

    @property
    def setpoint_pitch(self) -> np.ndarray:
        return self._pitch
