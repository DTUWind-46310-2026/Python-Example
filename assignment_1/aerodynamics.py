from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import Simulation

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import newaxis as na
from rotation import Rotation
from scipy.interpolate import RegularGridInterpolator
from timing import timer


class AerodynamicsBase(ABC):
    """
    Base (parent) class for aerodynamic calculations. This is not supposed to be used during the simulations (and also
    doesn't do anything). Using the @abstractmethod line defines which methods the children classes need to implement.

    Required methods are:
        - `step()`

    The method `simulation_init()` does nothing by default and can be overwritten (in the children).
    """

    def simulation_init(self, simulation: Simulation):
        pass

    @abstractmethod
    def step(self, simulation: Simulation):
        pass


class Aerodynamics(AerodynamicsBase):

    def __init__(
        self,
        polar_data_directory="data",
        glauert=True,
        prandtl=True,
        dynamic_wake=True,
        dynamic_stall=True,
        wake_effect: bool | str = True,
        rho=1.225,
        skip_last_blade_elements=1,
    ):
        """
        Initialises an instance used for aerodynamic calculations.

        Parameters
        ----------
        polar_data_directory : str, optional
            Directory containing polar data. The polar data must be defined in files named `rel_t_*.csv`, where `*` is
            the airfoil's relative thickness multiplied by 1000. The files must be csv files containing the columns
            `alpha,cl_stdy,cd_stdy,cm_stdy,f_s,cl_inv,cl_fs`. All files must have the same `alpha` values. By default
            "data".
        glauert : bool, optional
            Whether or not to use Glauert's correction for heavily loaded rotors, by default True.
        prandtl : bool, optional
            Whether or not to use Prandtl's correction for finite number of blades, by default True.
        dynamic_wake : bool, optional
            Whether or not to use Øye's dynamic wake model, by default True.
        dynamic_stall : bool, optional
            Whether or not to use Øye's dynamic stall model, by default True.
        wake_effect : bool | str, optional
            Whether or not to redistribute the induced velocities under yaw, by default True. When `wake_effect=True`,
            the geomtrical model is used. Accepted values are `True`, `False`, `geometrical`, `empirical`.
        rho : float, optional
            Air density, by default 1.225.
        skip_last_blade_elements : int, optional
            How many blade elements (defined by simulation.strucutre) to skip, counting from the tip. By default 1.
        """
        self._polar_data = self.load_polars(polar_data_directory)
        self.glauert = glauert
        self.prandtl = prandtl
        self.dynamic_wake = dynamic_wake
        self.dynamic_stall = dynamic_stall
        self.wake_effect = wake_effect
        self.rho = rho
        self.skip_last_blade_elements = skip_last_blade_elements
        self.k = 0.6

        self._polar_interpolant = {}
        self.a = np.zeros(0)
        self.inflow = np.zeros(0)
        self.W = np.zeros(0)
        self.W_int = np.zeros(0)
        self.W_qs = np.zeros(0)
        self.phi = np.zeros(0)
        self.alpha = np.zeros(0)
        self.V_rel_magnitude = np.zeros(0)
        self.f_s = np.zeros(0)
        self.section_lift = np.zeros(0)
        self.section_drag = np.zeros(0)
        self.wind1 = np.zeros(0)

        self.structure_data = {}
        self._wake_r_idx = 0

    def simulation_init(self, simulation: Simulation):
        # Keep some (constant!) structural data saved to the aerodynamics instance
        self.structure_data["n_blades"] = simulation.structure.n_blades
        self.structure_data["n_elements"] = simulation.structure.n_elements
        self.structure_data["r"] = simulation.structure.r
        self.structure_data["R"] = simulation.structure.R

        # Simulation.structure.n_elements != simulation.aerodynamics.n_elements
        self.n_elements = simulation.structure.n_elements - self.skip_last_blade_elements
        self.r = simulation.structure.r[: self.n_elements]
        self._wake_r_idx = np.argmax(self.r >= self.structure_data["R"] * 0.7)  # used for the geometrical wake effects

        # Set up the polar interpolants
        rthick = self._polar_data["rel_thickness"]
        alpha_vals = self._polar_data["alpha"]
        element_thicknesses = simulation.structure.rel_thickness[: self.n_elements]
        element_indices = np.arange(self.n_elements)

        ip_blade_thickness = np.repeat(element_thicknesses, len(alpha_vals))
        ip_blade_alpha = np.tile(alpha_vals, self.n_elements)
        ip_blade_polars = np.column_stack([ip_blade_thickness, ip_blade_alpha])
        for var, data in [(v, d) for v, d in self._polar_data.items() if v not in ("rel_thickness", "alpha")]:
            rgi_thickness = RegularGridInterpolator((rthick, alpha_vals), data)
            element_data = rgi_thickness(ip_blade_polars).reshape(self.n_elements, len(alpha_vals))
            self._polar_interpolant[var] = RegularGridInterpolator((element_indices, alpha_vals), element_data)
        self._element_indices = element_indices

        # Allocate memory for the aerodynamic calculations
        shape_1var = (self.structure_data["n_blades"], self.n_elements)
        shape_3vars = (self.structure_data["n_blades"], self.n_elements, 3)
        self.a = np.zeros(shape_1var)
        self.W = np.zeros(shape_3vars)
        self.W_int = np.zeros(shape_3vars)
        self.W_qs = np.zeros(shape_3vars)
        self.inflow = np.zeros(shape_3vars)
        self.V_rel = np.zeros(shape_3vars)
        self.phi = np.zeros(shape_1var)
        self.alpha = np.zeros(shape_1var)
        self.section_lift = np.zeros(shape_1var)
        self.section_drag = np.zeros(shape_1var)
        self.f_s = np.zeros(shape_1var)
        self.wind1 = np.zeros(shape_3vars)

    @timer
    def step(self, simulation: Simulation):
        self.step_inflow(simulation)
        self.step_forces(simulation)
        self.step_induction(simulation)

    @timer
    def step_inflow(self, simulation: Simulation):
        ids_blade = list(range(simulation.structure.n_blades))
        blades_pos = np.asarray([simulation.structure.blade_x1(i_blade)[: self.n_elements] for i_blade in ids_blade])
        wind1 = np.asarray([simulation.wind(blades_pos[i_blade]) for i_blade in ids_blade])
        wind5 = np.asanyarray([simulation.structure.x15(wind1[i_blade], i_blade) for i_blade in ids_blade])
        blades_vel5 = np.asarray([simulation.structure.blade_u5(i_blade)[: self.n_elements] for i_blade in ids_blade])

        self.wind1 = wind1
        self.inflow = wind5 - blades_vel5
        self.V_rel = self.inflow + self.W
        self.V_rel_magnitude = np.linalg.norm(self.V_rel[:, :, 1:], axis=2)
        self.phi = np.arctan(self.V_rel[:, :, 2] / -self.V_rel[:, :, 1])
        self.alpha = self.phi - simulation.structure.twist[: self.n_elements] - simulation.structure.pitch[:, na]

    @timer
    def step_forces(self, simulation: Simulation):
        n_blades = simulation.structure.n_blades
        interp_points = np.column_stack([np.tile(self._element_indices, n_blades), self.alpha.ravel()])
        cl_stdy_blades = self._polar_interpolant["cl_stdy"](interp_points).reshape(self.alpha.shape)
        cd_stdy_blades = self._polar_interpolant["cd_stdy"](interp_points).reshape(self.alpha.shape)

        cd = cd_stdy_blades
        if self.dynamic_stall:
            tau = 4 * simulation.structure.chord[: self.n_elements] / np.abs(self.V_rel_magnitude)
            f_s = self._polar_interpolant["f_s"](interp_points).reshape(self.alpha.shape)
            cl_inv = self._polar_interpolant["cl_inv"](interp_points).reshape(self.alpha.shape)
            cl_fs = self._polar_interpolant["cl_fs"](interp_points).reshape(self.alpha.shape)
            self.f_s = f_s + (self.f_s - f_s) * np.exp(-simulation.dt / tau)
            cl = cl_inv * self.f_s + cl_fs * (1 - self.f_s)
        else:
            cl = cl_stdy_blades

        q = 0.5 * self.rho * self.V_rel_magnitude**2
        self.section_lift = q * cl * simulation.structure.chord[: self.n_elements]
        self.section_drag = q * cd * simulation.structure.chord[: self.n_elements]

    @timer
    def step_induction(self, simulation: Simulation):
        n_blades = simulation.structure.n_blades
        R = simulation.structure.R

        self.a = -self.W[:, :, 2] / simulation.wind.hub_mean
        f_g = 1 if not self.glauert else np.where(self.a <= 1 / 3, 1, 0.25 * (5 - 3 * self.a))
        F = 1
        if self.prandtl:
            exp_arg = -n_blades * (R - self.r) / (2 * self.r * np.sin(np.abs(self.phi)))
            F = 2 / np.pi * np.arccos(np.exp(exp_arg))

        denom_velocity = np.sqrt(self.wind1[:, :, 1] ** 2 + (self.wind1[:, :, 2] + f_g * self.W[:, :, 2]) ** 2)
        W_qs_magnitude = -n_blades * self.section_lift / (4 * np.pi * self.rho * self.r * F * denom_velocity)
        W_qs = np.asarray(
            [np.zeros_like(self.phi), W_qs_magnitude * np.sin(self.phi), W_qs_magnitude * np.cos(self.phi)]
        )
        # Now W_qs is in shape (uvw, n_blades, n_elements) but needs to in in shape (n_blades, n_elements, uvw)
        W_qs = W_qs.transpose(1, 2, 0)

        if self.wake_effect and simulation.structure.yaw != 0:
            blade_azimuths = simulation.structure.blade_azimuth(np.asarray(range(simulation.structure.n_blades)))
            d_azi = blade_azimuths - simulation.structure.max_downstream_azimuth
            if self.wake_effect == "geometrical" or self.wake_effect is True:
                W_wake5 = self.W[:, self._wake_r_idx].mean(axis=0)
                W_wake2 = Rotation.rotate_3d_y(W_wake5, simulation.structure.tilt)
                W_wake1 = Rotation.rotate_3d_x(W_wake2, simulation.structure.yaw)
                V_wake = np.asarray([0, 0, simulation.wind.hub_mean]) + W_wake1
                chi = np.arccos(np.dot(simulation.structure.rotor_normal, V_wake) / np.linalg.norm(V_wake))
            elif self.wake_effect == "empirical":
                Ct = self.thrust / (0.5 * self.rho * np.pi * R**2 * simulation.wind.hub_mean**2)
                a_glob = 0.246 * Ct + 0.0586 * Ct**2 + 0.0883 * Ct**3
                chi = (0.6 * a_glob + 1) * simulation.structure.yaw
            else:
                raise NotImplementedError(f"{self.wake_effect=} but implemented are 'geometrical', 'empirical'.")
            W_qs *= 1 + self.r[na, :, na] / R * np.tan(chi / 2) * np.cos(d_azi[:, na, na])

        if self.dynamic_wake:
            tau_1 = 1.1 * R / ((1 - 1.3 * np.clip(self.a, None, 0.5)) * simulation.wind.hub_mean)
            tau_2 = tau_1 * (0.39 - 0.26 * (self.r / R) ** 2)

            H = W_qs + self.k * tau_1[:, :, na] * (W_qs - self.W_qs) / simulation.dt
            self.W_int = H + (self.W_int - H) * np.exp(-simulation.dt / tau_1[:, :, na])
            self.W = self.W_int + (self.W - self.W_int) * np.exp(-simulation.dt / tau_2[:, :, na])
            self.W_qs = W_qs
        else:
            self.W = W_qs

    @property
    def py(self):
        py_full = np.zeros((self.structure_data["n_blades"], self.structure_data["n_elements"]))
        py_full[:, : self.n_elements] = self.section_lift * np.sin(self.phi) - self.section_drag * np.cos(self.phi)
        return py_full

    @property
    def pz(self):
        pz_full = np.zeros((self.structure_data["n_blades"], self.structure_data["n_elements"]))
        pz_full[:, : self.n_elements] = self.section_lift * np.cos(self.phi) + self.section_drag * np.sin(self.phi)
        return pz_full

    @property
    def thrust(self):
        return np.trapezoid(self.pz, self.structure_data["r"], axis=1).sum()

    @property
    def torque(self):
        return np.trapezoid(self.py * self.structure_data["r"], self.structure_data["r"], axis=1).sum()

    def thrust_blade(self, blade_idx: int):
        return np.trapezoid(self.pz[blade_idx], self.structure_data["r"]).sum()

    def power(self, simulation: Simulation):
        return self.torque * simulation.structure.omega_shaft

    @staticmethod
    def load_polars(dir_polars: str | Path, filenames="rel_t_*.csv"):

        def get_thickness(filename):
            skip = filenames.find("*")
            until_character = filenames[skip + 1]
            return float(filename[skip : skip + filename[skip:].find(until_character)]) / 10

        polars = {}
        for file_polar in Path(dir_polars).glob(filenames):
            polars[get_thickness(file_polar.name)] = pd.read_csv(file_polar)

        variable_names = list([*polars.values()][0].columns)
        variable_names.pop(variable_names.index("alpha"))
        rthicknesses = sorted(polars.keys())
        alpha = np.deg2rad([*polars.values()][0]["alpha"].to_numpy())
        data = {"rel_thickness": np.array(rthicknesses), "alpha": alpha}
        for variable in variable_names:
            data[variable] = np.array([polars[t][variable].values for t in rthicknesses])
        return data


class NoAerodynamics(AerodynamicsBase):
    def simulation_init(self, simulation: Simulation):
        pass

    def step(self, simulation: Simulation):
        pass
