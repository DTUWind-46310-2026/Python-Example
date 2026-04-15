from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import Simulation

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from rotation import Rotation
from timing import timer


class StructureBase(ABC):
    """
    Base (parent) class for the structure. This is not supposed to be used during the simulations. Using the
    @abstractmethod line defines which methods the children classes need to implement. Required methods are:

    - `step()`
    - `blade_x1()`
    - `blade_u5()`
    - `x15()`
    - `x51()`

    This class defines some functionalities that are useful for the child classes (RigidStructure and at some
    point a flexible structure).
    """

    def __init__(
        self,
        omega_init: float,
        file_blade: str | Path,
        radius: float,
        hub_height: float,
        l_shaft: float,
        yaw: float,
        tilt: float,
        cone: float,
        pitch_init: tuple[float, ...],
        pitch_range: tuple[float, float],
        tower_yz: tuple[float, float],
        tower_radius: tuple[tuple[float, ...], tuple[float, ...]],
        rotor_inertia: float,
    ) -> None:
        """
        Sets up some instance variables for the child classes. Also defines
            - `max_downstream_azimuth`: azimuth at which a blade points furthest in the positive `z` direction
            - `rotor_normal`: array of length 1 pointing normal to the (unconed) rotor plane. If `yaw`, `tilt` are both
            zero, then `rotor_normal` points in the positive `z` direction.
        Both are automatically updated when `yaw`, `tilt` or `cone` changes.

        Parameters
        ----------
        omega_init : float
            The initial rotational speed of the rotor in rad/s
        file_blade : str or Path
            Path to the csv file defining the blade structure. Expected columns: `radius`, `chord`, `twist`,
            `rel_thickness`
        radius : float
            Rotor radius (tip radius) in metres
        hub_height : float
            Hub height of the wind turbine in metres
        l_shaft : float
            Length of the shaft in metres
        yaw : float
            Yaw angle of the rotor in degrees
        tilt : float
            Tilt angle of the shaft in degrees
        cone : float
            Coning angle of the rotor in degrees
        pitch_init : tuple[float, ...]
            Initial pitch angles (degrees) for each blade. The number of blades is inferred from the
            length of this tuple
        pitch_range : tuple[float, float]
            Allowed pitch range (min, max) in degrees
        tower_yz : tuple[float, float]
            The `(y, z)` position of the tower base in metres
        tower_radius : tuple[tuple[float, ...], tuple[float, ...]]
            The tower radius distribution over `x` defined as a tuple of tuples of `(x, radius)`.
        rotor_inertia : float
            Moment of inertia of the rotor about the shaft axis in kg·m²
        """
        df_blade_data = pd.read_csv(file_blade)
        r = df_blade_data["radius"].to_numpy()
        self.r = r
        self.R = radius
        self.r_hub = r[0]
        self.chord = df_blade_data["chord"].to_numpy()
        self.twist = np.deg2rad(df_blade_data["twist"].to_numpy())
        self.rel_thickness = df_blade_data["rel_thickness"].to_numpy()
        self.n_elements = r.size

        self.tower_yz = tower_yz
        self.tower_radius = np.asarray(tower_radius)
        self.rotor_inertia = rotor_inertia
        self.pitch_range = np.deg2rad(pitch_range)

        self.hub_height = hub_height
        self.l_shaft = l_shaft
        self._yaw = np.deg2rad(yaw)
        self._tilt = np.deg2rad(tilt)
        self._cone = np.deg2rad(cone)
        self.n_blades = len(pitch_init)
        self._pitch = np.deg2rad(pitch_init)
        self.max_downstream_azimuth = self._max_downstream_azimuth(self._yaw, self._tilt)
        self.rotor_normal = self._rotor_normal(self.yaw, self.tilt)

        self.azimuth_shaft = 0.0
        self.omega_shaft = omega_init

        self._x5_blade: np.ndarray = np.c_[self.r, np.zeros_like(self.r), np.zeros_like(self.r)]

    def simulation_init(self, simulation: Simulation):
        pass

    @abstractmethod
    def step(self, simulation: Simulation):
        pass

    @abstractmethod
    def blade_x1(self, blade_idx: int) -> np.ndarray:
        """
        Returns the coordinates of blade number `blade_idx` in the coordinate system 1.

        Parameters
        ----------
        blade_idx : int
            Index of blade.

        Returns
        -------
        np.ndarray
            The coordinates of the blade in coordinate system 1 as [x, y, z].
        """
        pass

    @abstractmethod
    def blade_u5(self, blade_idx: int) -> np.ndarray:
        """
        The velocities only due to the motion of the blade in the blade coordinate system.

        Parameters
        ----------
        blade_idx : int
            Blade index for which to get the velocities.

        Returns
        -------
        np.ndarray
            Velocities as numpy array as [u, v, w] in coordinate system 5.
        """
        pass

    @abstractmethod
    def x15(self, array: np.ndarray, blade_idx: int) -> np.ndarray:
        """
        Transforms an array from coordinate system 1 into the blade coordinate system 5.

        Parameters
        ----------
        array : np.ndarray
            The array with shape (n, 3) where each row is in the directions [x, y, z]
        blade_idx : int
            Blade index.

        Returns
        -------
        np.ndarray
            The transformed array in the blade coordinate system.
        """
        pass

    @abstractmethod
    def x51(self, array: np.ndarray, blade_idx: int) -> np.ndarray:
        """
        Transforms an array from coordinate system 5 into coordinate system 1.

        Parameters
        ----------
        array : np.ndarray
            The array with shape (n, 3) where each row is in the directions [x, y, z]
        blade_idx : int
            Blade index.

        Returns
        -------
        np.ndarray
            The transformed array in the coordinate system 1.
        """
        pass

    @property
    def yaw(self):
        return self._yaw

    @property
    def tilt(self):
        return self._tilt

    @property
    def cone(self):
        return self._cone

    @property
    def pitch(self):
        return self._pitch

    @yaw.setter
    def yaw(self, yaw):
        self._set_angle("_yaw", yaw)

    @cone.setter
    def cone(self, cone):
        self._set_angle("_cone", cone)

    @tilt.setter
    def tilt(self, tilt):
        self._set_angle("_tilt", tilt)

    def blade_azimuth(self, blade_idx):
        """
        Returns the azimuth angle of blade `blade_idx` in radians. Blades are equally spaced around
        the rotor; blade 0 has azimuth equal to `azimuth_shaft`.

        Parameters
        ----------
        blade_idx : int or array-like of int
            Index (or indices) of the blade(s). Must be less than `n_blades`.

        Returns
        -------
        float or np.ndarray
            Azimuth angle(s) in radians.

        Raises
        ------
        ValueError
            If any `blade_idx` exceeds `n_blades`.
        """
        if np.any(blade_idx > self.n_blades):
            raise ValueError(f"Structure only has '{self.n_blades}' blades, but {blade_idx=}.")
        return self.azimuth_shaft + blade_idx * 2 * np.pi / self.n_blades

    def _set_angle(self, angle_name: str, angle_value: float):
        """
        Set the angle `angle_name` of the instance to the value `np.deg2rad(value)`. Afterwards, update
        `max_downstream_azimuth` and `rotor_normal`.

        Parameters
        ----------
        angle_name : str
            Name of the angle attribute of the `StructureBase` instance.
        angle_value : float
            Angle in radians.
        """
        setattr(self, angle_name, np.deg2rad(angle_value))
        self.max_downstream_azimuth = self._max_downstream_azimuth(self.yaw, self.tilt)
        self.rotor_normal = self._rotor_normal(self.yaw, self.tilt)

    @staticmethod
    def _max_downstream_azimuth(yaw: float, tilt: float) -> float:
        if np.isclose(tilt, 0):  # Equation from the lecture doesn't hold for tilt=0.
            return np.pi / 2 if yaw >= 0 else -np.pi / 2
        return np.arctan(-np.tan(yaw) / (np.sin(tilt)))

    @staticmethod
    def _rotor_normal(yaw: float, tilt: float) -> np.ndarray:
        # Cone doesn't influence the rotor normal for the wake skew calculation
        normal4 = np.asarray([0, 0, 1])
        normal2 = Rotation.rotate_3d_y(normal4, tilt)
        return Rotation.rotate_3d_x(normal2, yaw)


class RigidStructure(StructureBase):

    def __init__(
        self,
        omega_init=0.0,
        file_blade="data/blade_data.csv",
        radius=89.17,
        hub_height=119.0,
        l_shaft=7.1,
        yaw=0.0,
        tilt=-5.0,
        cone=2.5,
        pitch_init: tuple[float, ...] = (-0.3196, -0.3196, -0.3196),
        pitch_range=(-0.3196, 90),
        tower_yz: tuple[float, float] = (0, 0),
        tower_radius: tuple[tuple[float, ...], tuple[float, ...]] = ((0, 119), (3.32, 3.32)),
        rotor_inertia=1.6e8,
        pitch_eigenfreq=8,
        pitch_damping_ratio=0.7,
    ) -> None:
        """
        Initialises an instance for a rigid wind turbine. See `StructureBase.__init__` for the
        full list of inherited parameters.

        Parameters
        ----------
        pitch_eigenfreq : float, optional
            Natural frequency of the pitch actuator model in rad/s, by default 8
        pitch_damping_ratio : float, optional
            Damping ratio of the pitch actuator model (dimensionless), by default 0.7
        """
        super().__init__(
            omega_init=omega_init,
            file_blade=file_blade,
            radius=radius,
            hub_height=hub_height,
            l_shaft=l_shaft,
            yaw=yaw,
            tilt=tilt,
            cone=cone,
            pitch_init=pitch_init,
            pitch_range=pitch_range,
            tower_yz=tower_yz,
            tower_radius=tower_radius,
            rotor_inertia=rotor_inertia,
        )
        self.pitch_eigenfreq = pitch_eigenfreq
        self.pitch_damping_ratio = pitch_damping_ratio
        self._previous_pitch = np.asarray(self.pitch)

    @timer
    def step(self, simulation: Simulation):
        """
        Advances the structure one time step.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
        torque_diff = simulation.aerodynamics.torque - simulation.controller.generator_torque
        self.omega_shaft += torque_diff / self.rotor_inertia * simulation.dt
        self.azimuth_shaft += self.omega_shaft * simulation.dt
        if not np.all(simulation.controller.setpoint_pitch == self.pitch):
            term_1 = (self.pitch_eigenfreq * simulation.dt) ** 2 * simulation.controller.setpoint_pitch
            term_2 = (2 - (self.pitch_eigenfreq * simulation.dt) ** 2) * self.pitch
            term_3 = (self.pitch_damping_ratio * self.pitch_eigenfreq * simulation.dt - 1) * self._previous_pitch
            term_4 = 1 + self.pitch_damping_ratio * self.pitch_eigenfreq * simulation.dt

            self._pitch, self._previous_pitch = (term_1 + term_2 + term_3) / term_4, self._pitch

    def blade_x1(self, blade_idx: int) -> np.ndarray:
        """
        Returns the coordinates of blade number `blade_idx` in the coordinate system 1.

        Parameters
        ----------
        blade_idx : int
            Index of blade.

        Returns
        -------
        np.ndarray
            The coordinates of the blade in coordinate system 1 as [x, y, z].
        """
        x4_blade = Rotation.rotate_3d_y(self._x5_blade, self.cone)
        x3_blade = Rotation.rotate_3d_z(x4_blade, self.blade_azimuth(blade_idx))
        x2_blade = Rotation.rotate_3d_y(x3_blade + np.asarray([0, 0, -self.l_shaft]), self.tilt)
        return Rotation.rotate_3d_x(x2_blade + np.asarray([self.hub_height, 0, 0]), self.yaw)

    def blade_u5(self, blade_idx: int) -> np.ndarray:
        """
        The velocities only due to the motion of the blade in the blade coordinate system.

        Parameters
        ----------
        blade_idx : int
            Blade index for which to get the velocities.

        Returns
        -------
        np.ndarray
            Velocities as numpy array as [u, v, w] in coordinate system 5.
        """
        return np.c_[np.zeros_like(self.r), self.omega_shaft * self.r, np.zeros_like(self.r)]

    def x15(self, array: np.ndarray, blade_idx: int) -> np.ndarray:
        """
        Transforms an array from coordinate system 1 into the blade coordinate system 5.

        Parameters
        ----------
        array : np.ndarray
            The array with shape (n, 3) where each row is in the directions [x, y, z]
        blade_idx : int
            Blade index.

        Returns
        -------
        np.ndarray
            The transformed array in the blade coordinate system.
        """
        x2 = Rotation.rotate_3d_x(array, -self.yaw)
        x3 = Rotation.rotate_3d_y(x2, -self.tilt)
        x4 = Rotation.rotate_3d_z(x3, -self.blade_azimuth(blade_idx))
        return Rotation.rotate_3d_y(x4, -self.cone)

    def x51(self, array: np.ndarray, blade_idx: int) -> np.ndarray:
        """
        Transforms an array from the blade coordinate system 5 into coordinate system 1.
        This is the inverse of `x15`.

        Parameters
        ----------
        array : np.ndarray
            Array with shape (n, 3) where each row is in the directions [x, y, z] of CS5.
        blade_idx : int
            Blade index.

        Returns
        -------
        np.ndarray
            The transformed array in coordinate system 1.
        """
        x4 = Rotation.rotate_3d_y(array, self.cone)
        x3 = Rotation.rotate_3d_z(x4, self.blade_azimuth(blade_idx))
        x2 = Rotation.rotate_3d_x(x3, self.tilt)
        return Rotation.rotate_3d_y(x2, self.yaw)


class PitchingRigidStructure(RigidStructure):

    def __init__(
        self,
        *steps: tuple[float, float],
        omega_init=0.0,
        file_blade="data/blade_data.csv",
        radius=89.17,
        hub_height=119.0,
        l_shaft=7.1,
        yaw=0.0,
        tilt=-5.0,
        cone=2.5,
        pitch_init: tuple[float, ...] = (-0.3196, -0.3196, -0.3196),
        pitch_range=(-0.3196, 90),
        tower_yz: tuple[float, float] = (0, 0),
        tower_radius: tuple[tuple[float, ...], tuple[float, ...]] = ((0, 119), (3.32, 3.32)),
        rotor_inertia=1.6e8,
    ) -> None:
        """
        A rigid structure that applies prescribed pitch step changes at specified times.
        See `RigidStructure.__init__` for all other accepted keyword arguments. The pitch is applied
        immediately; no pitch dynamics are modelled.

        Parameters
        ----------
        *steps : tuple[float, float]
            Any number of `(t_i, pitch_i)` pairs. When `simulation.time >= t_i`, the collective
            pitch of all blades is set to `pitch_i` degrees. Steps are applied in the order given
            and the last pitch value is held indefinitely.
        """
        super().__init__(
            omega_init=omega_init,
            file_blade=file_blade,
            radius=radius,
            hub_height=hub_height,
            l_shaft=l_shaft,
            yaw=yaw,
            tilt=tilt,
            cone=cone,
            pitch_init=pitch_init,
            pitch_range=pitch_range,
            tower_yz=tower_yz,
            tower_radius=tower_radius,
            rotor_inertia=rotor_inertia,
        )

        self._step_times = np.asarray([step[0] for step in steps])
        self._step_pitch = np.deg2rad(np.asarray([step[1] for step in steps] + [steps[-1][0] + 1e-5]))
        self._i_current_pitch = -1

    @timer
    def step(self, simulation: Simulation):
        # Adjust the pitch
        if (i := (np.argwhere(simulation.time >= self._step_times)[-1])) > self._i_current_pitch:
            self._i_current_pitch = i
            self._pitch = np.full_like(self._pitch, self._step_pitch[i])

        # Advance the rotor position
        super().step(simulation)
