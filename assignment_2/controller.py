from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import Simulation

import numpy as np


class ControllerBase(ABC):
    """
    Base (parent) class for wind turbine controllers. This is not supposed to be used directly in
    simulations. Using the @abstractmethod decorator defines which methods child classes must implement.

    Required methods/properties are:
        - `step()`
        - `generator_torque`
        - `setpoint_pitch`
        - `power()`
    """

    @abstractmethod
    def step(self, simulation: Simulation):
        pass

    def simulation_init(self, simulation: Simulation):
        pass

    @property
    @abstractmethod
    def generator_torque(self) -> float:
        """
        Current generator torque setpoint in N·m.
        """
        pass

    @property
    @abstractmethod
    def setpoint_pitch(self) -> np.ndarray:
        """
        Pitch setpoint for each blade in radians, as a 1-D array of length `n_blades`.
        """
        pass

    @abstractmethod
    def power(self, simulation: Simulation) -> float:
        """
        Returns the estimated electrical power output at the current time step.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.

        Returns
        -------
        float
            Electrical power in W.
        """
        pass


class PIController(ControllerBase):
    """
    Proportional-Integral (PI) collective pitch and generator-torque controller.

    Below rated speed, the generator torque follows a k-ω² law to track the optimal tip-speed
    ratio. Above rated speed, the generator torque holds constant power or constant torque
    (depending on `above_rated_mode`), while a PI pitch controller with gain scheduling keeps
    the rotor speed at `omega_ref`.
    """

    def __init__(self, Kp=1.5, Ki=0.64, KK=14, omega_ref_factor=1.0, above_rated_mode="power") -> None:
        """
        Initialises the PI controller.

        Parameters
        ----------
        Kp : float, optional
            Proportional gain of the pitch PI controller [rad/(rad/s)], by default 1.5.
        Ki : float, optional
            Integral gain of the pitch PI controller [rad/rad], by default 0.64.
        KK : float, optional
            Gain scheduling constant [deg]. The effective gain is reduced by the factor
            `GK = 1 / (1 + θ / KK)` where `θ` is the current pitch angle, by default 14.
        omega_ref_factor : float, optional
            Scales the rated rotor speed to set the speed reference `omega_ref`. Values below
            1.0 shift the below-/above-rated transition to a lower speed, by default 1.0.
        above_rated_mode : str, optional
            Generator torque strategy above rated speed. `"power"` holds constant power;
            `"torque"` holds constant torque, by default `"power"`.
        """
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
        """
        Caches quantities derived from the simulation that remain constant throughout the run:
        the optimal torque gain `K_omega_opt`, the rated generator torque, the speed reference
        `omega_ref`, and the initial pitch state.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
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
        """
        Updates the generator torque setpoint for the current time step.

        Below `omega_ref` the k-ω² law is applied. Above `omega_ref` the torque is set
        according to `above_rated_mode`: `"power"` for constant aerodynamic power or
        `"torque"` for constant rated torque.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
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
        """
        Step the collective pitch. The PI pitch controller here is technically always active, but it is basically turned
        off below rated by the pitch limits.

        Parameters
        ----------
        simulation : Simulation
            Simulation instance
        """
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

    def power(self, simulation: Simulation):
        return self.generator_torque * simulation.structure.omega_shaft
