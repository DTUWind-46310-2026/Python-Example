import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aerodynamics import Aerodynamics
from controller import ConstantRotSpeedController, PIController
from recorder import aero_power_recorder
from scipy.optimize import minimize
from simulation import Simulation
from structure import RigidStructure
from wind import ConstantWind, ShearWind, TurbulentWind, WindSteps

do = {
    "CP_optimisation": True,
    "task_1": True,
    "task_2": True,
}

plot = {
    "task_1": True,
    "task_2": True,
}

(dir_opti := Path("optimisation")).mkdir(exist_ok=True, parents=True)
(dir_task_1 := Path("task_1")).mkdir(exist_ok=True, parents=True)
(dir_task_2 := Path("task_2")).mkdir(exist_ok=True, parents=True)


if do["CP_optimisation"]:
    controller = ConstantRotSpeedController()
    wind = ConstantWind(11.44)
    aerodynamics = Aerodynamics(dynamic_wake=False, dynamic_stall=False)

    opti_history = []
    dt = 0.05

    def power(x):
        omega, pitch = x
        sim = Simulation(
            structure=RigidStructure(omega_init=omega, pitch_init=(pitch, pitch, pitch)),
            controller=controller,
            wind=wind,
            aerodynamics=aerodynamics,
            recorders=aero_power_recorder(),
            verbose=False,
        )
        sim.run(dt, 10)
        power = sim.get_recorders()["aero_power"]["aero_power"][-int(2 * np.pi / (3 * omega * dt)) :].mean()
        opti_history.append((*x, power))
        print(f"Step {len(opti_history)}, power: {round(power)}")
        return -power

    opti = minimize(power, x0=[0.8, 0.0], method="Nelder-Mead")
    cp = -opti.fun / (0.5 * aerodynamics.rho * np.pi * RigidStructure().R ** 2 * wind.hub_mean**3)
    tsr = opti.x[0] * RigidStructure().R / wind.hub_mean
    v_rated = (10.64e6 / (0.5 * aerodynamics.rho * np.pi * RigidStructure().R ** 2 * cp)) ** (1 / 3)
    omega_rated = tsr * v_rated / RigidStructure().R
    with open(dir_opti / "optimised.json", "w") as f:
        json.dump(
            {
                "CP": cp,
                "TSR": tsr,
                "pitch": opti.x[1],
                "v_rated": v_rated,
                "omega_rated": omega_rated,
            },
            f,
            indent=4,
        )

    history = np.asarray(opti_history)
    pd.DataFrame({"TSR": history[:, 0], "pitch": history[:, 1], "power": history[:, 2]}).to_csv(
        dir_opti / "optimisation_history.csv", index=False
    )


if do["task_1"]:
    base_wind = ConstantWind(1)
    from recorder import aero_power_recorder, pitch_recorder

    T_below_rated = 180
    T_above_rated = 40
    wind_steps = [(T_below_rated * i - 4 * T_below_rated, i) for i in range(4, 12)] + [
        (T_above_rated * i + 5 * T_below_rated, i) for i in range(12, 26)
    ]
    aero = Aerodynamics()
    structure = RigidStructure(7.8052 * 4 / 89.17)
    sim = Simulation(
        wind=WindSteps(base_wind, *wind_steps),
        structure=structure,
        aerodynamics=aero,
        controller=PIController(),
        recorders=[aero_power_recorder(), pitch_recorder()],
    )
    sim.run(0.1, wind_steps[-1][0] + T_above_rated, dir_task_1, True)

    data = sim.get_recorders()
    CP = np.zeros(len(wind_steps))
    pitch = np.zeros(len(wind_steps))
    ws = np.asarray(wind_steps)
    for i, (t_step, ws_step) in enumerate(wind_steps[1:]):
        idx = (t_step <= data["time"]).argmax() - 1
        CP[i] = data["aero_power"]["aero_power"][idx] / (aero.rho * 0.5 * np.pi * structure.R**2 * (ws_step - 1) ** 3)
        pitch[i] = data["pitch"]["pitch"][idx]
    CP[-1] = data["aero_power"]["aero_power"][-1] / (aero.rho * 0.5 * np.pi * structure.R**2 * 25**3)
    pitch[-1] = data["pitch"]["pitch"][-1]
    pd.DataFrame({"ws": ws[:, 1], "CP": CP, "pitch": pitch}).to_csv(dir_task_1 / "CP_and_pitch.csv", index=False)

if plot["task_1"]:
    df_report_pitch = pd.read_csv("data/report_pitch_rpm.csv")
    df_report_cp = pd.read_csv("data/report_power_thrust.csv")

    cp_and_pitch = pd.read_csv(dir_task_1 / "CP_and_pitch.csv")

    fig, ax = plt.subplots()
    ax.plot(df_report_pitch["ws"], df_report_pitch["pitch"], label="Report")
    ax.plot(cp_and_pitch["ws"], cp_and_pitch["pitch"], label="Own")
    ax.legend()
    fig.savefig(dir_task_1 / "pitch.pdf")

    fig, ax = plt.subplots()
    ax.plot(df_report_cp["ws"], df_report_cp["CP"], label="Report")
    ax.plot(cp_and_pitch["ws"], cp_and_pitch["CP"], label="Own")
    ax.legend()
    fig.savefig(dir_task_1 / "CP.pdf")


if do["task_2"]:
    hub_mean = 11.4
    shear_wind = ShearWind(119, hub_mean, 0.2)

    turb_file = dir_task_2 / "turb.nc"
    if not turb_file.is_file():
        turb_wind = TurbulentWind.generate((1024, 64, 64), (20, 4, 4), 0.1, shear_wind, turb_file, hub_mean)
    else:
        turb_wind = TurbulentWind.load(turb_file, shear_wind)

    from recorder import (
        aero_power_recorder,
        aero_torque_recorder,
        controller_mode,
        generator_power_recorder,
        generator_torque_recorder,
        pitch_recorder,
        setpoint_pitch_recorder,
        thrust_recorder,
    )

    aero = Aerodynamics()
    structure = RigidStructure(1.002)
    sim = Simulation(
        wind=turb_wind,
        structure=structure,
        aerodynamics=aero,
        controller=PIController(),
        recorders=[
            aero_power_recorder(),
            aero_torque_recorder(),
            controller_mode(),
            generator_power_recorder(),
            generator_torque_recorder(),
            pitch_recorder(),
            setpoint_pitch_recorder(),
            thrust_recorder(),
        ],
    )
    sim.run(0.1, 1200, dir_task_2, True)

if plot["task_2"]:
    data = {res.stem: pd.read_csv(res) for res in dir_task_2.glob("*.csv")}

    fig, ax = plt.subplots()
    data["aero_power"].plot.line(x="time", ax=ax)
    data["gen_power"].plot.line(x="time", ax=ax)
    fig.savefig(dir_task_2 / "power.pdf")

    fig, ax = plt.subplots()
    data["pitch"].plot.line(x="time", ax=ax)
    data["sp_pitch"].plot.line(x="time", ax=ax)
    fig.savefig(dir_task_2 / "pitch.pdf")

    fig, ax = plt.subplots()
    data["aero_torque"].plot.line(x="time", ax=ax)
    data["generator_torque"].plot.line(x="time", ax=ax)
    fig.savefig(dir_task_2 / "torque.pdf")

    fig, ax = plt.subplots()
    data["controller_modes"].plot.line(x="time", ax=ax)
    fig.savefig(dir_task_2 / "controller_modes.pdf")
    plt.show()
