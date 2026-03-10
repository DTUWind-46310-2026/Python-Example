from pathlib import Path

import numpy as np
import pandas as pd
from aerodynamics import AerodynamicsBase, NoAerodynamics
from recorder import Recorder, time_recorder
from structure import StructureBase
from timing import timer
from tqdm import trange
from wind import NoWind, WindBase


class Simulation:

    def __init__(
        self,
        structure: StructureBase,
        aerodynamics: AerodynamicsBase = NoAerodynamics(),
        wind: WindBase = NoWind(),
        recorders: Recorder | list[Recorder] | None = None,
    ) -> None:
        """
        Creates a simulation instance.

        Parameters
        ----------
        structure : Structure
            The structure instance.
        wind : Wind, optional
            The wind instance., by default NoWind()
        recorders : Recorder | list[Recorder] | None, optional
            Any number of recorders. By default, a recorder is added that saves the times of the simulation.
        """
        self.structure = structure
        self.wind = wind
        self.aerodynamics = aerodynamics
        self.model_parts = [self.wind, self.aerodynamics, self.structure]
        self.time = 0
        self.dt = 0
        self.step_idx = 0

        recorders = recorders or []
        self.recorders = recorders if isinstance(recorders, list) else [recorders]
        self.recorders.append(time_recorder())
        self.timer = timer

    def run(self, dt: float, T: float, dir_out: str | Path | None = None, overwrite: bool = False):
        """
        Run the simulation.

        Parameters
        ----------
        dt : float
            Time step duration.
        T : float
            Time the simulation runs for.
        dir_out : str | Path | None, optional
            Directory to save recorder CSVs and timing report to. By default None (no saving).
        overwrite : bool, optional
            Whether to overwrite existing files, by default False
        """
        self.timer.reset()
        self.dt = dt

        n_sim_steps = int(T / dt) + 1
        for recorder in self.recorders:
            recorder.update_n_steps(n_sim_steps)

        for part in self.model_parts:
            part.simulation_init(self)

        for step_idx in trange(n_sim_steps):
            self.step_idx = step_idx
            self.time = round(self.dt * step_idx, 4)

            for part in self.model_parts:
                part.step(self)

            for recorder in self.recorders:
                recorder(self)

        if dir_out is not None:
            self.save_recorders(dir_out, overwrite=overwrite)
            self.timer.report(Path(dir_out) / "timing.json", T, dt)

    def get_recorders(self) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """
        Returns the data of all the recorders.

        Example
        -------
        If you added a recorder `Recorder(record_function, "my_recorder", ("u", "v", "w"))` to the simulation, then the
        return of `get_recorders()` will be
        ```
        {"time": <time at each time step>,
         "my_recorder": {
            "u": <u time series>,
            "v": <v time series>,
            "w": <w time series>,
            }
         }
        ```
        Adding more recorders to the simulation adds more keys with the recorders' names and their data (as
        dictionaries again) to the returned dictionary.

        Returns
        -------
        dict[str, np.ndarray | dict[str, np.ndarray]]
            Dictionary with format {<recorder_name>: {<quantity name>: <quantity data>} | {"time": <times of simulation>}
        """
        data = {rec.name: {dim: rec.data[:, i] for i, dim in enumerate(rec.func_returns)} for rec in self.recorders}
        data["time"] = data["time"].pop("time")
        return data

    def save_recorders(self, root: str | Path, case_name="", overwrite=False):
        """
        Save the data of the recorders to files in the `root` directory. The files will have the names
        `<recorder_name><case_name>.csv`. The file headers are `"time"` and the names specified by `func_returns`
        when defining each recorder.

        Parameters
        ----------
        root : str | Path
            The directory into which the files will be saved.
        case_name : str, optional
            What to append to the file name., by default ""
        overwrite : bool, optional
            Whether or not to overwrite if the file exists already, by default False
        """
        recorders = self.get_recorders()
        time = {"time": recorders.pop("time")}
        (_r := Path(root)).mkdir(parents=True, exist_ok=True)
        for rec_name, data in recorders.items():
            if (save_to := (_r / (rec_name + f"{case_name}.csv"))).is_file() and not overwrite:
                print(f"Skipping '{save_to.as_posix()}' because it already exists and 'overwrite=False'")
                continue
            pd.DataFrame(time | data).to_csv(save_to, index=False)
