from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import Simulation

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from rotation import Rotation
from time_integration import NewmarkIntegrator
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
        Initialises an instance for a rigid wind turbine. See `StructureBase` for more information.

        Parameters
        ----------
        drive_train_dynamics : bool, optional
            Whether or not to include drive train dynamics, by default False
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
        self._advance_pitch_actuator(simulation)

    def _advance_pitch_actuator(self, simulation: Simulation):
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
        tower_yz: tuple[float, float] = (0, 0),
        tower_radius: tuple[tuple[float, ...], tuple[float, ...]] = ((0, 119), (3.32, 3.32)),
    ) -> None:
        """
        See `RigidStructure` for more information. `*steps` can be any number of tuples defining (t_i, pitch_i), i.e.
        for `simulation.time >= t_i`, `pitch_i` is applied.
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
            tower_radius=tower_radius,
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


class ElasticStructure(RigidStructure):
    """
    Wind turbine with elastic tower fore-aft and flexibile or stiff blades. The degrees of
    freedom of the instance attribute `q` are:

      idx  0: x_t                          (tower fore-aft displacement)
      idx  1: phi_shaft                    (shaft azimuth)
      idx  2 + 3*b .. 4 + 3*b: q_1..q_3    (1st flap, 1st edge, 2nd flap)
                                           for each flexible blade
                                           b in [0, n_flexible_blades).

    The flexibility of the tower and blades only affect the motion-induced velocities
    for the blade aerodynamics. The displacements are calculated as well, but they do not influence
    aerodynamic calculations (such as sampling the turbulence box at a different point).

    The equations of motion are integrated in time by a Newmark-beta integrator with iterative residual correction.
    """

    def __init__(
        self,
        omega_init=0.1,
        file_blade="data/blade_data.csv",
        file_modes="data/blade_mode_shapes.csv",
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
        rotor_inertia=0.0,
        pitch_eigenfreq=8,
        pitch_damping_ratio=0.7,
        mode_freqs: tuple[float, float, float] = (3.93, 6.10, 11.28),
        log_decrements: tuple[float, float, float] = (0.03, 0.03, 0.03),
        tower_mass: float = 446e3,
        tower_stiffness: float = 1.7e6,
        tower_damping_ratio: float = 0.01,
        gravity: float = 9.81,
        newmark_gamma: float = 0.51,
        newmark_beta: float = 0.25,
        newmark_eps: float = 1e-6,
        n_flexible_blades: int = 3,
    ) -> None:
        """
        See the parent class hierachy for parameters definitions of those parameters that are not described here.

        Parameters
        ----------
        file_modes : str, optional
            File defining the 3 mode shapes. Must have columns r,u1fy,u1fz,u1ey,u1ez,u2fy,u2fz,m, where r is the radial
            distance along the blade, u<i><j><k> describes mode number i of the j blade-direction (flap or edge) in the
            blade coorindate system k, by default "data/blade_mode_shapes.csv"
        rotor_inertia : float, optional
            Inertia of the shaft and generator. The rotational inertia due to the blades is always included., by
            default 0.0
        mode_freqs : tuple[float, float, float], optional
            Frequencies of the three modes in rad/s, by default (3.93, 6.10, 11.28)
        log_decrements : tuple[float, float, float], optional
            Logarithmic decrements of the three modes, by default (0.03, 0.03, 0.03)
        tower_mass : float, optional
            Mass of the tower top, hub, generator, etc. Must not include the blade masses. by default 446e3
        tower_stiffness : float, optional
            Equivalent stiffness of the tower in the fore-aft direction, by default 1.7e6
        tower_damping_ratio : float, optional
            Logarithmic damping ratio of the tower fore-aft motion, by default 0.01
        gravity : float, optional
            Gravitational acceleration in m/s², by default 9.81
        newmark_gamma : float, optional
            Newmark's gamma , by default 0.51
        newmark_beta : float, optional
            Newmark's beta, by default 0.25
        newmark_eps : float, optional
            Residual until which the Newmark algorithm iteratores, by default 1e-6
        n_flexible_blades : int, optional
            Number of flexible blades. The total number of blades (stiff + flexible) is defined through the number of
            values in `pitch_init`, by default 3
        """

        self.n_blades = len(pitch_init)
        if not 0 <= n_flexible_blades <= self.n_blades:
            raise ValueError(f"n_flexible_blades must be in [0, {self.n_blades}], got {n_flexible_blades}")
        self.n_flexible_blades = n_flexible_blades
        self.n_dof = 2 + 3 * self.n_flexible_blades

        # these need to be before super().__init__()
        self._q = np.zeros(self.n_dof)
        self._q_dot = np.zeros(self.n_dof)
        self._q_ddot = np.zeros(self.n_dof)

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
            pitch_eigenfreq=pitch_eigenfreq,
            pitch_damping_ratio=pitch_damping_ratio,
        )

        self.mode_freqs = np.asarray(mode_freqs)
        self.log_decrements = np.asarray(log_decrements)
        self.tower_mass = tower_mass
        self.tower_stiffness = tower_stiffness
        self.tower_damping_ratio = tower_damping_ratio
        self.gravity = gravity

        df_modes = pd.read_csv(file_modes)
        if not np.allclose(df_modes["r"].to_numpy(), self.r):
            raise ValueError(f"Mode-shape radii in {file_modes} do not match blade radii from {file_blade}.")

        # Zero-pitch mode shapes, shape (3 modes, n_elements)
        self._phi_y_org = np.vstack(
            [df_modes["u1fy"].to_numpy(), df_modes["u1ey"].to_numpy(), df_modes["u2fy"].to_numpy()]
        )
        self._phi_z_org = np.vstack(
            [df_modes["u1fz"].to_numpy(), df_modes["u1ez"].to_numpy(), df_modes["u2fz"].to_numpy()]
        )
        self._blade_mass = df_modes["m"].to_numpy()

        self._integrator = NewmarkIntegrator(gamma=newmark_gamma, beta=newmark_beta, eps=newmark_eps)

        self._M = np.zeros((self.n_dof, self.n_dof))
        self._C = np.zeros((self.n_dof, self.n_dof))
        self._K = np.zeros((self.n_dof, self.n_dof))
        self._last_pitch = 1e9

    @property
    def omega_shaft(self) -> float:
        return self._q_dot[1]

    @omega_shaft.setter
    def omega_shaft(self, value: float) -> None:
        self._q_dot[1] = value

    @property
    def azimuth_shaft(self) -> float:
        return self._q[1]

    @azimuth_shaft.setter
    def azimuth_shaft(self, value: float) -> None:
        self._q[1] = value

    @property
    def q(self) -> np.ndarray:
        return self._q

    @property
    def q_dot(self) -> np.ndarray:
        return self._q_dot

    @property
    def q_ddot(self) -> np.ndarray:
        return self._q_ddot

    def _rotated_modes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Rotate the zero-pitch mode shapes by the collective blade pitch.
        Returns (phi_y, phi_z) each of shape (n_blades, n_modes, n_elements).
        """
        cos_p = np.cos(self.pitch)[:, None, None]  # (n_blades, 1, 1)
        sin_p = np.sin(self.pitch)[:, None, None]
        phi_y_org = self._phi_y_org[None, :, :]  # (1, n_modes, n_elements)
        phi_z_org = self._phi_z_org[None, :, :]
        phi_y = phi_z_org * sin_p + phi_y_org * cos_p
        phi_z = phi_z_org * cos_p - phi_y_org * sin_p
        return phi_y, phi_z

    @timer
    def _assemble_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cos_p = np.cos(self.pitch[0])  #! Assumes collective pitch
        sin_p = np.sin(self.pitch[0])
        phi_y = self._phi_z_org * sin_p + self._phi_y_org * cos_p  # (n_modes, n_elements)
        phi_z = self._phi_z_org * cos_p - self._phi_y_org * sin_p

        r = self.r
        m = self._blade_mass
        m_blade = np.trapezoid(m, r)

        GM = np.trapezoid(m * (phi_y**2 + phi_z**2), r, axis=1)  # (n_modes,)
        S_z = np.trapezoid(m * phi_z, r, axis=1)
        S_y = np.trapezoid(m * r * phi_y, r, axis=1)

        n_dof = self.n_dof
        M = np.zeros((n_dof, n_dof))
        M[0, 0] = self.tower_mass + self.n_blades * m_blade
        M[1, 1] = self.rotor_inertia + self.n_blades * np.trapezoid(m * r**2, r)
        for b in range(self.n_flexible_blades):
            base = 2 + 3 * b
            for k in range(3):
                M[base + k, base + k] = GM[k]
                M[base + k, 0] = S_z[k]
                M[base + k, 1] = S_y[k]
        M = M + np.tril(M, -1).T  # Mirror to the upper triangle

        K = np.zeros((n_dof, n_dof))
        K[0, 0] = self.tower_stiffness
        for b in range(self.n_flexible_blades):
            base = 2 + 3 * b
            for k in range(3):
                K[base + k, base + k] = self.mode_freqs[k] ** 2 * GM[k]

        C = np.zeros((n_dof, n_dof))
        C[0, 0] = 2 * self.tower_damping_ratio * np.sqrt(self.tower_stiffness / M[0, 0]) * M[0, 0]
        for b in range(self.n_flexible_blades):
            base = 2 + 3 * b
            for k in range(3):
                C[base + k, base + k] = self.log_decrements[k] / np.pi * self.mode_freqs[k] * GM[k]

        return M, K, C

    def _gravity_in_blade_frame(self, blade_idx: int) -> tuple[float, float]:
        """
        Project gravity onto the blade local (y, z) directions for the given blade.
        """
        g_global = np.array([[-self.gravity, 0.0, 0.0]])
        g_blade5 = self.x15(g_global, blade_idx)[0]
        return float(g_blade5[1]), float(g_blade5[2])  # components are constant along blade

    @timer
    def _compute_Q(self, simulation: Simulation) -> np.ndarray:
        Q = np.zeros(self.n_dof)
        aero = simulation.aerodynamics
        Q[0] = aero.thrust
        Q[1] = aero.torque - simulation.controller.generator_torque

        phi_y, phi_z = self._rotated_modes()
        r = self.r
        m = self._blade_mass

        for b in range(self.n_flexible_blades):
            g_y, g_z = self._gravity_in_blade_frame(b)
            for k in range(3):
                Q_aero = np.trapezoid(aero.py[b] * phi_y[b, k] + aero.pz[b] * phi_z[b, k], r)
                Q_grav = np.trapezoid(m * (g_y * phi_y[b, k] + g_z * phi_z[b, k]), r)
                Q[2 + 3 * b + k] = Q_aero + Q_grav
        return Q

    @timer
    def step(self, simulation: Simulation):
        if not np.isclose(self._last_pitch, self.pitch[0]):
            self._M, self._K, self._C = self._assemble_matrices()
            self._last_pitch = self.pitch[0]

        Q = self._compute_Q(simulation)

        self._q, self._q_dot, self._q_ddot = self._integrator.step(
            self._q, self._q_dot, self._q_ddot, self._M, self._K, self._C, Q, simulation.dt
        )

        self._advance_pitch_actuator(simulation)

    def eigen_analysis(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        Solve the generalised eigenvalue problem (K^-1) M GX = (1/omega^2) GX
        for the assembled mass and stiffness matrices, after dropping the
        azimuth row/column (index 1) which has no stiffness.

        Returns
        -------
        freqs : np.ndarray
            Natural frequencies in Hz, sorted ascending.
        modes : np.ndarray
            Mode shapes (columns), sorted to match `freqs`. Rows correspond
            to the kept DOF indices (`keep`).
        keep : list[int]
            DOF indices kept (i.e. all except the azimuth DOF).
        """
        M, K, _ = self._assemble_matrices()
        keep = [i for i in range(self.n_dof) if i != 1]
        M_r = M[np.ix_(keep, keep)]
        K_r = K[np.ix_(keep, keep)]
        lam, vec = np.linalg.eig(np.linalg.solve(K_r, M_r))
        omega = 1.0 / np.sqrt(np.real(lam))
        order = np.argsort(omega)
        modes = np.real(vec[:, order])
        # Normalise each mode shape so its largest-magnitude entry is +1
        col = np.arange(modes.shape[1])
        idx_max = np.argmax(np.abs(modes), axis=0)
        modes = modes / modes[idx_max, col]
        return omega[order] / (2 * np.pi), modes, keep

    def blade_u5(self, blade_idx: int) -> np.ndarray:
        # Rigid rotational contribution: [0, omega * r, 0]
        omega = self._q_dot[1]
        u5_rigid = np.c_[np.zeros_like(self.r), omega * self.r, np.zeros_like(self.r)]

        # Tower fore-aft velocity (along global x) transformed into blade frame 5.
        # Same for every element.
        x_dot_global = np.array([[self._q_dot[0], 0.0, 0.0]])
        u5_tower_row = self.x15(x_dot_global, blade_idx)[0]
        u5_tower = np.broadcast_to(u5_tower_row, u5_rigid.shape)

        if blade_idx >= self.n_flexible_blades:
            return u5_rigid + u5_tower

        # Modal velocity: sum over modes of phi_k(r) * q_dot_k(b)
        phi_y, phi_z = self._rotated_modes()
        base = 2 + 3 * blade_idx
        q_dot_blade = self._q_dot[base : base + 3]  # (n_modes,)
        u5_modal = np.zeros_like(u5_rigid)
        u5_modal[:, 1] = phi_y[blade_idx].T @ q_dot_blade
        u5_modal[:, 2] = phi_z[blade_idx].T @ q_dot_blade

        return u5_rigid + u5_tower + u5_modal
