from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import Simulation

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from timing import timer


class WindBase(ABC):
    """
    Base (parent) class for the wind. This is not supposed to be used during the simulations (and also doesn't do
    anything). Using the @abstractmethod line defines which methods the children classes need to implement.
    """

    def simulation_init(self, simulation: Simulation):
        pass

    @abstractmethod
    def __call__(self, xyz) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, simulation: Simulation) -> None:
        pass

    @property
    @abstractmethod
    def hub_mean(self) -> float:
        pass


class ConstantWind(WindBase):
    def __init__(self, ws: float) -> None:
        """
        Initialises a wind instance that returns a constant wind speed everywhere.

        Parameters
        ----------
        ws : float
            The wind speed.
        """
        self.ws = ws

    @timer
    def __call__(self, xyz):
        xyz = np.atleast_2d(xyz)
        return (np.c_[np.zeros_like(xyz[:, 0]), np.zeros_like(xyz[:, 0]), np.full_like(xyz[:, 0], self.ws)]).squeeze()

    @timer
    def step(self, simulation: Simulation):
        # Nothing needs to happen here; the wind speed simply stays constant everywhere.
        pass

    @property
    def hub_mean(self):
        return self.ws


class NoWind(ConstantWind):
    def __init__(self) -> None:
        """
        Initialises an instance that returns a wind speed of zero everywhere.
        """
        super().__init__(0)


class ShearWind(WindBase):
    def __init__(self, x_ref: float, v_ref: float, exponent: float) -> None:
        """
        Initialises an instance that returns wind speeds based on the defined shear. `hub_mean` is defined as the wind
        wind speed of the shear at the hub height of the tower (received during `simulation_init`).

        Parameters
        ----------
        x_ref : float
            The x coordinate at which the reference wind speed `v_ref` is defined.
        v_ref : float
            The reference wind speed at height `x_ref`
        exponent : float
            The exponent of for the shear.
        """
        self.shear = lambda x: v_ref * (x / x_ref) ** exponent
        self._hub_mean: float = None

    def simulation_init(self, simulation: Simulation):
        self._hub_mean = self.shear(simulation.structure.hub_height)

    @timer
    def __call__(self, xyz):
        xyz = np.atleast_2d(xyz)
        return np.c_[np.zeros_like(xyz[:, 0]), np.zeros_like(xyz[:, 0]), self.shear(xyz[:, 0])].squeeze()

    @timer
    def step(self, simulation: Simulation):
        # Nothing needs to happen here either.
        pass

    @property
    def hub_mean(self):
        return self._hub_mean


class WindWithTower(WindBase):

    def __init__(self, surrounding_wind: WindBase) -> None:
        """
        Initialises and instance that returns wind speeds based on `surrounding_wind` including the
        tower effect. The tower position and radius distribution is taken from `simulation.structure` during
        the `simulation_init()`. The `hub_mean` is taken from `surrounding_wind`.

        Example
        ----------
        To use a shear with `x_ref=119`, `u_ref=10`, `exponent=0.2` that includes the tower effect:

        >>> shear_wind = ShearWind(119, 10, 0.2)
        >>> shear_with_tower = WindWithTower(shear_wind)

        Parameters
        ----------
        surrounding_wind : WindBase
            An instance of a wind class that has the `WindBase` class as parent class.
        """
        self.surrounding_wind = surrounding_wind
        self._tower_yz = np.zeros(0)

    def simulation_init(self, simulation: Simulation):
        self.surrounding_wind.simulation_init(simulation)

        self._tower_yz = simulation.structure.tower_yz
        self.a = interp1d(
            simulation.structure.tower_radius[:, 0],
            simulation.structure.tower_radius[:, 1],
            fill_value=(0, 0),
            bounds_error=False,
        )

    @timer
    def __call__(self, xyz):
        xyz = np.atleast_2d(xyz)

        x, y, z = xyz.T
        r = np.linalg.norm((xyz[:, 1:3] - self._tower_yz), axis=1)
        V_0 = np.atleast_2d(self.surrounding_wind(xyz))[:, 2]
        v_r = z / r * V_0 * (1 - (self.a(x) / r) ** 2)
        v_theta = y / r * V_0 * (1 + (self.a(x) / r) ** 2)

        v_y = y / r * v_r - z / r * v_theta
        v_z = z / r * v_r + y / r * v_theta
        return (np.c_[np.zeros_like(v_y), v_y, v_z]).squeeze()

    @timer
    def step(self, simulation: Simulation):
        # Nothing needs to change here :)
        pass

    @property
    def hub_mean(self):
        return self.surrounding_wind.hub_mean


try:
    import hipersim
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    class TurbulentWind(WindBase):

        def __init__(self, mann_turbulence_field: xr.DataArray, mean_wind: WindBase) -> None:
            """
            Initialises a turbulent wind instance. NOT to be used directly. Rather use
                - `turb_wind = TurbulentWind.generate(...)` to create a new turbulent wind instance
                - `turb_wind = TurbulentWind.load(...)` to load an existing turbulent wind file

            The wind speed returned by `turb_wind()` is the sum of the mean wind of `mean_wind` and the turbulent
            fluctuations of `turb_wind`. `turb_wind.turb_mean` and `turb_wind.turb_std` store the mean and standard
            deviation of the `u`, `v`, and `w` components of the turbulent fluctuations.

            `turb_wind.periodic` can be set to `True` (default) or `False` to enable/disable periodicity in `z`.
            `hub_mean` is taken from `mean_wind`.

            Parameters
            ----------
            mann_turbulence_field : xr.DataArray
                DataArray of turbulent wind fluctuations as created by `hipersim.MannTurbulenceField.generate()` but
                adjusted by the code in `TurbulentWind.generate()`.
            mean_wind : WindBase
                The mean wind.
            """
            self._mtf = mann_turbulence_field
            self.mean_wind = mean_wind
            self.z_advected = 0
            self.turb_mean = (
                float(self._mtf.sel(uvw="u").mean()),
                float(self._mtf.sel(uvw="v").mean()),
                float(self._mtf.sel(uvw="w").mean()),
            )
            self.turb_std = (
                float(self._mtf.sel(uvw="u").std()),
                float(self._mtf.sel(uvw="v").std()),
                float(self._mtf.sel(uvw="w").std()),
            )

            self.periodic = False
            self._box_length = float(self._mtf["z"][-1])

        @property
        def hub_mean(self) -> float:
            return self.mean_wind.hub_mean

        def simulation_init(self, simulation: Simulation):
            self.mean_wind.simulation_init(simulation)

            # Move the turbulence box to be centred on the hub in `x` and `y` and move the end of the box (in `z`) to
            # be at the front of the turbine.
            x_box = self._mtf["x"].values
            y_box = self._mtf["y"].values
            z_box = self._mtf["z"].values

            tower_top_pos = np.asarray([simulation.structure.hub_height] + list(simulation.structure.tower_yz))
            x_center = float((x_box[0] + x_box[-1]) / 2)
            y_center = float((y_box[0] + y_box[-1]) / 2)

            simulation.structure.azimuth_shaft = simulation.structure.max_downstream_azimuth
            z_max_blade = simulation.structure.blade_x1(0)[-1, 2]
            simulation.structure.azimuth_shaft = 0

            x_shifted = x_box - x_center + tower_top_pos[0]
            y_shifted = y_box - y_center + tower_top_pos[1]
            z_shifted = z_box - z_box[-1] + tower_top_pos[2] + z_max_blade
            self._mtf = self._mtf.assign_coords(x=x_shifted, y=y_shifted, z=z_shifted)
            self._z_min = float(z_shifted[0])

            # Create interpolants that are used in `__call__()`
            self._turb_interpolants = {
                component: RegularGridInterpolator(
                    (x_shifted, y_shifted, z_shifted), self._mtf.sel(uvw=component).values
                )
                for component in "uvw"
            }

        @timer
        def step(self, simulation: Simulation) -> None:
            self.z_advected = self.hub_mean * simulation.time

        @timer
        def __call__(self, xyz) -> np.ndarray:
            xyz = np.atleast_2d(xyz)
            # The moving of the box is given by `self.z_advected`
            pts_z = xyz[:, 2] - self.z_advected
            if self.periodic:
                pts_z = self._z_min + (pts_z - self._z_min) % self._box_length
            pts = np.c_[xyz[:, 0], xyz[:, 1], pts_z]
            turb_fluctuations = np.column_stack([self._turb_interpolants[c](pts) for c in ["u", "v", "w"]]).squeeze()
            return self.mean_wind(xyz) + turb_fluctuations

        @staticmethod
        def generate(
            Nxyz: tuple[int, int, int],
            dxyz: tuple[float, float, float],
            TI: float,
            mean_wind: WindBase,
            save: str | Path = "",
            hub_mean: float | None = None,
        ) -> TurbulentWind:
            """
            Generate turbulent fluctuations based on `Nxyz`, `dxyz`, `TI`, and `mean_wind`. If `save` is given, save the fluctuations to a netcdf file (`save` has to end in `.nc`.).

            Parameters
            ----------
            Nxyz : tuple[int, int, int]
                Number of grid points in (`x`, `y`, `z`).
            dxyz : tuple[float, float, float]
                Distance between grid points in (`x`, `y`, `z`).
            TI : float
                Turbulence intensity related to `mean_wind.hub_mean`.
            mean_wind : WindBase
                The mean wind.
            save : str | Path, optional
                File path to where the turbulence fluctuations are saved, by default not saved.

            Returns
            -------
            TurbulentWind
                An instance of `TurbulentWind`.
            """

            # Generate the turbulence fluctuations
            mtf = hipersim.MannTurbulenceField.generate(Nxyz=Nxyz, dxyz=dxyz)
            # Scale the fluctuations
            hm = mean_wind.hub_mean if mean_wind.hub_mean is not None else hub_mean
            if hm is None:
                raise ValueError(
                    f"Need to specify 'hub_mean' in 'generate' when using wind '{mean_wind.__class__.__name__}'"
                )
            mtf.scale_TI(TI, hm)

            # Adjust the coordinate system
            mtf = mtf.to_xarray()
            mtf = mtf.rename({"x": "z", "z": "x"})
            mtf = mtf.assign_coords(y=-mtf["y"])
            mtf = xr.concat(
                [
                    mtf.sel(uvw="w").transpose(),
                    -mtf.sel(uvw="v").transpose(),
                    mtf.sel(uvw="u").transpose(),
                ],
                dim=xr.DataArray(["u", "v", "w"], dims="uvw"),
            )
            if save != "":
                mtf.to_netcdf(save)
            return TurbulentWind(mtf, mean_wind)

        @staticmethod
        def load(file: str | Path, mean_wind: WindBase) -> TurbulentWind:
            """
            Load turbulence fluctuations created by `TurbulentWind.generate()`. Note: Must use a `mean_wind` with the same `hub_mean` as that used for the `generate()` call. Otherwise, the turublence intensity will be off.

            Parameters
            ----------
            file : str | Path
                Path to the turbulence fluctuations.
            mean_wind : WindBase
                Mean wind.

            Returns
            -------
            TurbulentWind
                An instance of `TurbulentWind`.
            """
            return TurbulentWind(xr.open_dataarray(file), mean_wind)

except ImportError:
    print("class `TurbulentWind` not loaded since `hipersim` or `wetb` are not installed.")
