from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aerodynamics import Aerodynamics
from recorder import (
    induction_recorder,
    power_recorder,
    py_recorder,
    pz_recorder,
    thrust_recorder,
)
from scipy.signal import welch
from simulation import Simulation
from structure import PitchingRigidStructure, RigidStructure
from wind import ConstantWind, ShearWind, TurbulentWind

simulate = {
    "1": True,
    "2": True,
    "3": True,
    "4": True,
}

plot = {
    "1": True,
    "2": True,
    "3": True,
    "4": True,
}

(dir_1 := Path("task_1")).mkdir(exist_ok=True, parents=True)
(dir_2 := Path("task_2")).mkdir(exist_ok=True, parents=True)
(dir_3 := Path("task_3")).mkdir(exist_ok=True, parents=True)
(dir_4 := Path("task_4")).mkdir(exist_ok=True, parents=True)

if simulate["1"]:
    wind = ConstantWind(8)
    sim = Simulation(
        RigidStructure(0.72),
        Aerodynamics(),
        wind,
        [power_recorder(), thrust_recorder(), py_recorder(), pz_recorder()],
    )
    sim.run(0.05, 200, dir_1, True, "_unsteady")

    sim = Simulation(
        RigidStructure(0.72),
        Aerodynamics(dynamic_stall=False, dynamic_wake=False),
        wind,
        [power_recorder(), thrust_recorder(), py_recorder(), pz_recorder()],
    )
    sim.run(0.1, 5, dir_1, True, "_steady")

if plot["1"]:
    r = pd.read_csv("data/blade_data.csv")["radius"].to_numpy()

    df_py_us = pd.read_csv(dir_1 / "py_blade_0_unsteady.csv")
    df_pz_us = pd.read_csv(dir_1 / "pz_blade_0_unsteady.csv")
    df_power_us = pd.read_csv(dir_1 / "power_unsteady.csv")
    df_thrust_us = pd.read_csv(dir_1 / "thrust_unsteady.csv")

    df_py_steady = pd.read_csv(dir_1 / "py_blade_0_steady.csv")
    df_pz_steady = pd.read_csv(dir_1 / "pz_blade_0_steady.csv")
    df_power_steady = pd.read_csv(dir_1 / "power_steady.csv")
    df_thrust_steady = pd.read_csv(dir_1 / "thrust_steady.csv")

    for drop_for in [df_py_us, df_pz_us, df_py_steady, df_pz_steady]:
        drop_for.drop(columns="time", inplace=True)

    # Load distributions at last time step
    fig, ax = plt.subplots()
    ax.plot(r, df_py_us.iloc[-1], label="$p_y$ unsteady")
    ax.plot(r, df_py_steady.iloc[-1], "--", label="$p_y$ steady")

    ax.plot(r, df_pz_us.iloc[-1], label="$p_z$ unsteady")
    ax.plot(r, df_pz_steady.iloc[-1], "--", label="$p_z$ steady")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Loads (N/m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(dir_1 / "load_distributions.pdf")

    # # Power and thrust time series
    fig, ax = plt.subplots()
    df_power_us.plot.line(x="time", ax=ax, label="unsteady")
    ax.hlines(df_power_steady["power"].iloc[-1], 0, df_power_us["time"].iloc[-1], "k", "--", label="steady")
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    plt.tight_layout()
    plt.savefig(dir_1 / "power.pdf")

    fig, ax = plt.subplots()
    df_thrust_us.plot.line(x="time", ax=ax, label="unsteady")
    ax.hlines(df_thrust_steady["thrust"].iloc[-1], 0, df_thrust_us["time"].iloc[-1], "k", "--", label="steady")
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thrust (N)")
    plt.tight_layout()
    plt.savefig(dir_1 / "thrust.pdf")
    plt.show()


if simulate["2"]:
    wind = ShearWind(119, 8, 0.2)
    sim = Simulation(
        RigidStructure(0.72),
        Aerodynamics(),
        wind,
        [
            power_recorder(),
            thrust_recorder(),
            thrust_recorder(0),
            thrust_recorder(1),
            thrust_recorder(2),
        ],
    )
    sim.run(0.05, 200, dir_2, True)

if plot["2"]:
    fig, ax = plt.subplots()
    df_power = pd.read_csv(dir_2 / "power.csv")
    df_power.plot.line(x="time", ax=ax)
    fig.savefig(dir_2 / "power.pdf", bbox_inches="tight")

    fig, ax = plt.subplots()
    for blade_idx in range(3):
        name = f"thrust_blade_{blade_idx}"
        pd.read_csv(dir_2 / f"{name}.csv").plot.line(x="time", y=name, ax=ax, label=f"Blade {blade_idx}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thrust (N)")
    pd.read_csv(dir_2 / "thrust.csv").plot.line(x="time", ax=ax)
    plt.show()


if simulate["3"]:
    wind = ConstantWind(8)
    sim = Simulation(
        PitchingRigidStructure((0, 0), (100, 2), (150, 0), omega_init=0.72),
        Aerodynamics(),
        wind,
        [
            power_recorder(),
            thrust_recorder(),
            induction_recorder(0, 10),
        ],
    )
    sim.run(0.05, 200, dir_3, True)

if plot["3"]:
    fig, ax = plt.subplots()
    pd.read_csv(dir_3 / "power.csv").plot.line("time", ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (w)")
    fig.savefig(dir_3 / "power.pdf", bbox_inches="tight")

    fig, ax = plt.subplots()
    pd.read_csv(dir_3 / "thrust.csv").plot.line("time", ax=ax)
    ax.set_ylabel("Thrust (N)")
    ax.set_xlabel("Time (s)")
    fig.savefig(dir_3 / "thrust.pdf", bbox_inches="tight")

    fig, ax = plt.subplots()
    pd.read_csv(dir_3 / "induction.csv").plot.line("time", ax=ax)
    ax.set_ylabel("Induction (m/s)")
    ax.set_xlabel("Time (s)")
    fig.savefig(dir_3 / "induction.pdf", bbox_inches="tight")

    plt.show()


dt_4 = 0.05
omega_4 = 0.72
if simulate["4"]:
    from wind import ShearWind, WindWithTower

    turb_file = dir_4 / "turb_field.nc"
    overwrite_box = False
    mean_wind = ConstantWind(8)
    mean_wind = ShearWind(119, 8, 0.2)
    if not turb_file.is_file() or overwrite_box:
        wind = TurbulentWind.generate((5016, 32, 32), (5, 6, 6), 0.1, mean_wind, turb_file)
    else:
        wind = TurbulentWind.load(turb_file, mean_wind)
    wind = WindWithTower(wind)
    sim = Simulation(
        RigidStructure(omega_4, yaw=5),
        Aerodynamics(),
        wind,
        # [
        #     thrust_recorder(),
        #     pz_recorder(),
        # ],
    )
    sim.run(dt_4, 3000, dir_4, True)


if plot["4"]:
    fig, ax = plt.subplots()
    df_thrust = pd.read_csv(dir_4 / "thrust.csv")
    df_thrust.plot.line("time", ax=ax)
    ax.set_ylabel("Thrust (N)")
    ax.set_xlabel("Time (s)")
    fig.savefig(dir_4 / "thrust.pdf", bbox_inches="tight")

    df_pz = pd.read_csv(dir_4 / "pz_blade_0.csv")
    fig, ax = plt.subplots()
    df_pz.plot.line("time", 8, ax=ax)
    ax.set_ylabel("pz at 65.75m")
    ax.set_xlabel("Time (s)")
    fig.savefig(dir_4 / "pz_11.pdf", bbox_inches="tight")

    def psd_fig(data: np.ndarray):
        data = data - data.mean()
        fig, ax = plt.subplots()
        f, Pxx = welch(data, fs=1 / dt_4, nperseg=4096)
        f_rotor = f / (omega_4 / (2 * np.pi))
        ax.semilogy(f_rotor, Pxx)
        ax.set_xlim(0, 10)
        ax.set_xlabel("Multiples of one rotor revolution")
        return fig, ax

    fig, ax = psd_fig(df_pz["8"].to_numpy())
    ax.set_ylabel("PSD of pz at r=65.75m ((N/m)²/Hz)")
    ax.set_xlabel("Time (s)")
    fig.savefig(dir_4 / "pz_11_psd.pdf", bbox_inches="tight")

    fig, ax = psd_fig(df_thrust["thrust"].to_numpy())
    ax.set_ylabel("PSD of total thrust (N²/Hz)")
    ax.set_xlabel("Time (s)")
    fig.savefig(dir_4 / "thrust_psd.pdf", bbox_inches="tight")

    plt.show()
