from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import Simulation

from abc import ABC, abstractmethod

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

    The method `simulation_init()` does nothing by default and can be overwritten (in the children).

    This class defines some functionalities that are useful for the child classes (RigidStructure and at some
    point a flexible structure).
    """

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
        pitch_init: tuple[float, ...] = (0, 0, 0),
        tower_yz: tuple[float, float] = (0, 0),
        tower_radius: tuple[tuple[float, ...], tuple[float, ...]] = ((0, 119), (3.32, 3.32)),
    ) -> None:
        """
        Sets up some instance variables for the child classes. Also defines
            - `max_downstream_azimuth`: azimuth at which a blade points furthest in the positive `z` direction
            - `rotor_normal`: array of length 1 pointing normal to the (unconed) rotor plane. If `yaw`, `tilt` are both
            zero, then `rotor_normal` points in the positive `z` direction.
        Both are automatically updated when `yaw`, `tilt` or `cone` changes.

        Parameters
        ----------
        omega_init : float, optional
            The initial rotational speed of the rotor, by default 0.0
        file_blade : str, optional
            Path to the file defining the blade structure. The path is expected to be a csv file
            with columns `r,c,twist,rel_thickness` for the radial position `r`, chord `c`, twist `twist`, and
            relative thickness `rel_thickness`, by default "data/blade_data.csv"
        hub_height : float, optional
            Hub height of the wind turbine, by default 119.0
        l_shaft : float, optional
            Length of the shaft, by default 7.1
        yaw : float, optional
            Yaw of the rotor, by default 0.0
        tilt : float, optional
            Tilt of the shaft, by default 0.0
        cone : float, optional
            Coning of the rotor, by default 0.0
        pitch_init : list, optional
            The initial pitch angles for each blade. From this, the number of blades are defined, by default (0, 0, 0)
        tower_zy : tuple[float, float], optional
            The `(y, z)` position of the tower base, by default (0, 0)
        tower_radius : tuple[tuple[float, ...], tuple[float, ...]]
            The tower radius distribution over `x` defined as `(x coords, radii)`, where `x coords` and `radii` are
            tuples of values with corresponding indices. By default `((0, 119), (3.32, 3.32))`, ie. constant radius of
            3.32m from 0m to 119m
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

        self.hub_height = hub_height
        self.l_shaft = l_shaft
        self._yaw = np.deg2rad(yaw)
        self._tilt = np.deg2rad(tilt)
        self._cone = np.deg2rad(cone)
        self.n_blades = len(pitch_init)
        self.pitch = np.deg2rad(pitch_init)
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

    @property
    def yaw(self):
        return self._yaw

    @property
    def tilt(self):
        return self._tilt

    @property
    def cone(self):
        return self._cone

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
        pitch_init: tuple[float, ...] = (0, 0, 0),
        tower_yz: tuple[float, float] = (0, 0),
        tower_radius: tuple[tuple[float, ...], tuple[float, ...]] = ((0, 119), (3.32, 3.32)),
        drive_train_dynamics=False,
    ) -> None:
        """
        Initialises an instance for a rigid wind turbine. See `StructureBase` for more information.

        Parameters
        ----------
        drive_train_dynamics : bool, optional
            Whether or not to include drive train dynamics, by default False
        """
        super().__init__(
            omega_init, file_blade, radius, hub_height, l_shaft, yaw, tilt, cone, pitch_init, tower_yz, tower_radius
        )

        self.drive_train_dynamics = drive_train_dynamics

    @timer
    def step(self, simulation: Simulation):
        """
        Advances the structure one time step.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.

        Raises
        ------
        NotImplementedError
            Drive train dynamics are not yet implemented.
        """
        if self.drive_train_dynamics:
            raise NotImplementedError("You'll have to implement the drive train dynamcis at some point :)")
        self.azimuth_shaft += self.omega_shaft * simulation.dt

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
        pitch_init: tuple[float, ...] = (0, 0, 0),
        tower_yz: tuple[float, float] = (0, 0),
        tower_radius: tuple[tuple[float, ...], tuple[float, ...]] = ((0, 119), (3.32, 3.32)),
        drive_train_dynamics=False,
    ) -> None:
        """
        See `RigidStructure` for more information. `*steps` can be any number of tuples defining (t_i, pitch_i), i.e.
        for `simulation.time >= t_i`, `pitch_i` is applied.

        Parameters
        ----------
        """
        super().__init__(
            omega_init,
            file_blade,
            radius,
            hub_height,
            l_shaft,
            yaw,
            tilt,
            cone,
            pitch_init,
            tower_yz,
            tower_radius,
            drive_train_dynamics,
        )

        self._step_times = np.asarray([step[0] for step in steps])
        self._step_pitch = np.deg2rad(np.asarray([step[1] for step in steps] + [steps[-1][0] + 1e-5]))
        self._i_current_pitch = -1

    @timer
    def step(self, simulation: Simulation):
        # Adjust the pitch
        if (i := (np.argwhere(simulation.time >= self._step_times)[-1])) > self._i_current_pitch:
            self._i_current_pitch = i
            self.pitch = np.full_like(self.pitch, self._step_pitch[self._i_current_pitch])

        # Advance the rotor position
        super().step(simulation)
