"""
Recorder class for storing time-series data during simulation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulation import Simulation


class Recorder:
    """
    Records simulation data in a pre-allocated numpy array.

    The recorder allocates memory based on dt and T.
    """

    def __init__(self, func: Callable, name: str, func_returns: tuple[str, ...] | str):
        """
        Create a recorder instance.

        Example
        ----------
        Wanted: Position of blade element at index 10 in coordinate system 1.
        First: Write a function that receives `simulation` and returns the position:
        >>> def get_blade_pos_in_1(simulation: Simulation):
        >>>     return simulation.structure.blade_x1(blade_idx=0)[10]

        Then: Create the recorder
        >>> pos_recorder = Recorder(get_blade_pos_in_1, "position_recorder", ("x", "y", "z"))

        Where "position_recorder" becomes the name of the recorder (when you use `simulation.get_recorders()`) and
        `("x", "y", "z")` are the coordinates that the `get_blade_pos_in_1()` returns.

        This example is already implemented as the `BladePosition1Recorder`.

        Parameters
        ----------
        func : Callable
            A function that receives only `simulation` as input and returns a 1D list or 1D numpy array of values.
        name : str
            The name for the recorded data. Important when using `simulation.get_recorders()`.
        func_returns : tuple[str, ...] | str
            Specify what the `func` returns, i.e., if it returns a xyz position, `func_returns = ("x", "y", "z")`.
        """
        self.func = func
        self.name = name
        self.func_returns = func_returns if isinstance(func_returns, tuple) else (func_returns,)
        self._data = np.empty(0)
        self._steps_udpated = False

    def update_n_steps(self, n_steps: int):
        self._data = np.zeros((n_steps, len(self.func_returns)))
        self._steps_udpated = True

    def __call__(self, simulation: Simulation):
        if not self._steps_udpated:
            raise RuntimeError(f"Need to use `update_n_steps` before using the recorder '{self.name}'.")
        self._data[simulation.step_idx] = self.func(simulation)

    @property
    def data(self) -> np.ndarray:
        return self._data


def time_recorder():
    def time(simulation: Simulation):
        return simulation.time

    return Recorder(time, "time", ("time",))


def blade_position_1_recorder(name: str, blade_idx: int, element_idx: int):
    def blade_pos(simulation: Simulation):
        return simulation.structure.blade_x1(blade_idx)[element_idx]

    return Recorder(blade_pos, name, ("x", "y", "z"))


def blade_velocity_5_recorder(name: str, blade_idx: int, element_idx: int | None = None):
    def blade_rel_vel(simulation: Simulation):
        vel5 = simulation.structure.blade_u5(blade_idx)[element_idx]

        blade_pos1 = simulation.structure.blade_x1(blade_idx)[element_idx]
        wind1 = simulation.wind(blade_pos1)
        wind5 = simulation.structure.x15(wind1, blade_idx)
        return vel5 + wind5

    return Recorder(blade_rel_vel, name, ("u", "v", "w"))


def wind_5_recorder(name: str, blade_idx: int, element_idx: int):
    def wind5(simulation: Simulation):
        blade_pos1 = simulation.structure.blade_x1(blade_idx)[element_idx]
        wind1 = simulation.wind(blade_pos1)
        return simulation.structure.x15(wind1, blade_idx)

    return Recorder(wind5, name, ("u", "v", "w"))


def py_recorder(blade_idx=0, n_elements=18):
    def py(simulation: Simulation):
        return simulation.aerodynamics.py[blade_idx]

    return Recorder(py, f"py_blade_{blade_idx}", tuple(map(str, range(n_elements))))


def pz_recorder(blade_idx=0, n_elements=18):
    def pz(simulation: Simulation):
        return simulation.aerodynamics.pz[blade_idx]

    return Recorder(pz, f"pz_blade_{blade_idx}", tuple(map(str, range(n_elements))))


def aero_power_recorder():
    def get_power(simulation: Simulation):
        return simulation.aerodynamics.power(simulation)

    return Recorder(get_power, "aero_power", "aero_power")


def aero_torque_recorder():

    def get_torque(simulation: Simulation):
        return simulation.aerodynamics.torque

    return Recorder(get_torque, "aero_torque", ("torque",))


def thrust_recorder(blade_idx: int | None = None):
    if blade_idx is None:

        def get_thrust(simulation: Simulation):
            return simulation.aerodynamics.thrust

    else:

        def get_thrust(simulation: Simulation):
            return simulation.aerodynamics.thrust_blade(blade_idx)

    name = "thrust" if blade_idx is None else f"thrust_blade_{blade_idx}"
    return Recorder(get_thrust, name, (name,))


def induction_recorder(blade_idx: int, element_idx: int):
    def induction(simulation: Simulation):
        return simulation.aerodynamics.W[blade_idx, element_idx]

    return Recorder(induction, "induction", ("wx", "wy", "wz"))


def rotation_speed_recorder():
    def rot_speed(simulation: Simulation):
        return simulation.structure.omega_shaft

    return Recorder(rot_speed, "rot_speed", "omega")


def generator_torque_recorder():
    def rot_speed(simulation: Simulation):
        return simulation.controller.generator_torque

    return Recorder(rot_speed, "generator_torque", "torque")


def pitch_recorder():
    def pitch(simulation: Simulation):
        return np.rad2deg(simulation.structure.pitch[0])

    return Recorder(pitch, "pitch", "pitch")


def setpoint_pitch_recorder():

    def sp_pitch(simulation: Simulation):
        return np.rad2deg(simulation.controller.setpoint_pitch[0])

    return Recorder(sp_pitch, "sp_pitch", "sp_pitch")


def generator_power_recorder():

    def gen_power(simulation: Simulation):
        return simulation.controller.power(simulation)

    return Recorder(gen_power, "gen_power", "gen_power")


def tsr_recorder():
    def tsr(simulation: Simulation):
        return simulation.structure.omega_shaft * simulation.structure.R / simulation.wind.hub_mean

    return Recorder(tsr, "tsr", "tsr")


def controller_mode():
    def modes(simulation: Simulation):
        return simulation.controller.modes

    return Recorder(modes, "controller_modes", ("generator", "pitch"))
