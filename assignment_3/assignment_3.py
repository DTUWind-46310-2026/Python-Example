from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aerodynamics import Aerodynamics
from animate_structure import animate_modes
from controller import PIController
from recorder import (
    flexible_DOFs_recorder,
    pitch_recorder,
    root_bending_moment_recorder,
    rotation_speed_recorder,
    tip_deflection_recorder,
    tower_position_recorder,
)
from scipy.signal import welch
from simulation import Simulation
from structure import ElasticStructure
from wind import ConstantWind, ShearWind, TurbulentWind, WindWithTower

do = {
    "task_1": True,
    "task_2": True,
    "task_3_eigen": True,
    "task_3_sims": True,
}

plot = {
    "task_1": True,
    "task_2": True,
    "task_3": True,
    "animate_system_modes": True,
}

(dir_task_1 := Path("task_1")).mkdir(exist_ok=True, parents=True)
(dir_task_2 := Path("task_2")).mkdir(exist_ok=True, parents=True)
(dir_task_3 := Path("task_3")).mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

R_ROTOR = 89.17
TSR_OPT = 7.8052
TI = 0.10


def omega_init(v0: float) -> float:
    """Reasonable starting rotor speed: TSR_opt below rated, rated above."""
    omega_rated = 1.002
    return min(TSR_OPT * v0 / R_ROTOR, omega_rated)


def make_turb(v0: float, T: float, file: Path) -> TurbulentWind:
    if file.is_file():
        return TurbulentWind.load(file, ConstantWind(v0))
    return TurbulentWind.generate((1024, 32, 32), (v0 * (T + 10) / 1024, 6, 6), TI, ConstantWind(v0), file, hub_mean=v0)


def base_recorders(blade_idx: int = 0, n_flexible_blades: int = 1):
    rec = [
        tower_position_recorder(),
        root_bending_moment_recorder(blade_idx),
        flexible_DOFs_recorder(n_flexible_blades),
        rotation_speed_recorder(),
        pitch_recorder(),
    ]
    rec += [tip_deflection_recorder(b) for b in range(n_flexible_blades)]
    return rec


def psd(signal: np.ndarray, dt: float, nperseg: int = 4096):
    signal = signal - signal.mean()
    nps = min(nperseg, signal.size)
    f, P = welch(signal, fs=1 / dt, nperseg=nps)
    return f, P


def add_rotor_marks(ax, omega_mean: float):
    f1p = omega_mean / (2 * np.pi)
    ax.set_xlim(0, 8 * f1p)
    for k, label in [(1, "1P"), (2, "2P"), (3, "3P"), (6, "6P")]:
        ax.axvline(k * f1p, color="k", lw=0.6, ls="--")
        ax.text(k * f1p, ax.get_ylim()[1], label, ha="center", va="top", fontsize=8)


# ---------------------------------------------------------------------------
# Q1 — 5 DOF, turbulent inflow at 7 and 18 m/s
# ---------------------------------------------------------------------------

q1_cases = {"7": 7.0, "18": 18.0}
dt_q1 = 0.05
T_q1 = 600.0

if do["task_1"]:
    for tag, v0 in q1_cases.items():
        case_dir = dir_task_1 / f"v{tag}"
        case_dir.mkdir(exist_ok=True, parents=True)
        turb = make_turb(v0, T_q1, case_dir / "turb.nc")
        sim = Simulation(
            structure=ElasticStructure(omega_init=omega_init(v0), tilt=0, cone=0, n_flexible_blades=1),
            controller=PIController(),
            aerodynamics=Aerodynamics(),
            wind=turb,
            recorders=base_recorders(blade_idx=0, n_flexible_blades=1),
        )
        sim.run(dt_q1, T_q1, case_dir, overwrite=True)


if plot["task_1"]:
    for tag in q1_cases:
        case_dir = dir_task_1 / f"v{tag}"
        if not case_dir.is_dir():
            continue
        df_tip = pd.read_csv(case_dir / "tip_deflections_0.csv")
        df_tower = pd.read_csv(case_dir / "tower_position.csv")
        df_root = pd.read_csv(case_dir / "root_bending_blade_0.csv")
        df_omega = pd.read_csv(case_dir / "rot_speed.csv")
        omega_mean = df_omega["omega"].iloc[len(df_omega) // 2 :].mean()

        # Time series
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(df_tip["time"], df_tip["flapwise"], label="flap")
        axes[0].plot(df_tip["time"], df_tip["edgewise"], label="edge")
        axes[0].set_ylabel("Tip deflection (m)")
        axes[0].legend()
        axes[1].plot(df_tower["time"], df_tower["x_t"])
        axes[1].set_ylabel("Tower disp. (m)")
        axes[2].plot(df_root["time"], df_root["flapwise"], label="flap")
        axes[2].plot(df_root["time"], df_root["edgewise"], label="edge")
        axes[2].set_ylabel("Root bending @ r=2.8 m (N·m)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()
        fig.suptitle(f"Q1 — turbulent V0={tag} m/s")
        fig.tight_layout()
        fig.savefig(case_dir / "time_series.pdf")

        # PSDs
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        signals = [
            (df_tip["flapwise"].to_numpy(), "Tip flap"),
            (df_tip["edgewise"].to_numpy(), "Tip edge"),
            (df_tower["x_t"].to_numpy(), "Tower disp."),
            (df_root["flapwise"].to_numpy(), "Root flap moment"),
        ]
        for ax, (sig, label) in zip(axes.ravel(), signals):
            f, P = psd(sig, dt_q1)
            ax.semilogy(f, P)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(f"PSD of {label}")
            add_rotor_marks(ax, omega_mean)
        fig.suptitle(f"Q1 — PSDs, V0={tag} m/s, omega_mean={omega_mean:.2f} rad/s")
        fig.tight_layout()
        fig.savefig(case_dir / "psd.pdf")


# ---------------------------------------------------------------------------
# Q2 — 5 DOF, steady shear inflow at 7 m/s
# ---------------------------------------------------------------------------

dt_q2 = 0.05
T_q2 = 300.0

if do["task_2"]:
    v0 = 7.0
    sim = Simulation(
        structure=ElasticStructure(omega_init=omega_init(v0), tilt=0, cone=0, n_flexible_blades=1),
        controller=PIController(),
        aerodynamics=Aerodynamics(),
        wind=ShearWind(119, v0, 0.2),
        recorders=base_recorders(blade_idx=0, n_flexible_blades=1),
    )
    sim.run(dt_q2, T_q2, dir_task_2, overwrite=True)


if plot["task_2"]:
    df_tip_s = pd.read_csv(dir_task_2 / "tip_deflections_0.csv")
    df_tower_s = pd.read_csv(dir_task_2 / "tower_position.csv")
    df_root_s = pd.read_csv(dir_task_2 / "root_bending_blade_0.csv")
    df_omega_s = pd.read_csv(dir_task_2 / "rot_speed.csv")
    omega_mean_s = df_omega_s["omega"].iloc[len(df_omega_s) // 2 :].mean()

    turb_dir = dir_task_1 / "v7"
    if (turb_dir / "tip_deflections_0.csv").is_file():
        df_tip_t = pd.read_csv(turb_dir / "tip_deflections_0.csv")
        df_tower_t = pd.read_csv(turb_dir / "tower_position.csv")
        df_root_t = pd.read_csv(turb_dir / "root_bending_blade_0.csv")
    else:
        df_tip_t = df_tower_t = df_root_t = None

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        (
            df_tip_s["flapwise"].to_numpy(),
            df_tip_t["flapwise"].to_numpy() if df_tip_t is not None else None,
            "Tip flap",
        ),
        (
            df_tip_s["edgewise"].to_numpy(),
            df_tip_t["edgewise"].to_numpy() if df_tip_t is not None else None,
            "Tip edge",
        ),
        (df_tower_s["x_t"].to_numpy(), df_tower_t["x_t"].to_numpy() if df_tower_t is not None else None, "Tower disp."),
        (
            df_root_s["flapwise"].to_numpy(),
            df_root_t["flapwise"].to_numpy() if df_root_t is not None else None,
            "Root flap moment",
        ),
    ]
    for ax, (sig_s, sig_t, label) in zip(axes.ravel(), panels):
        f, P = psd(sig_s, dt_q2)
        ax.semilogy(f, P, label="shear (Q2)")
        if sig_t is not None:
            f, P = psd(sig_t, dt_q1)
            ax.semilogy(f, P, label="turbulent (Q1)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"PSD of {label}")
        ax.legend()
        add_rotor_marks(ax, omega_mean_s)
    fig.suptitle("Q2 — PSDs at V0 = 7 m/s")
    fig.tight_layout()
    fig.savefig(dir_task_2 / "psd_compare.pdf")


# ---------------------------------------------------------------------------
# Q3 — 11 DOF system: eigenanalysis + simulations
# ---------------------------------------------------------------------------

dt_q3 = 0.05
T_q3 = 600.0
T_q3_steady = 200.0

if do["task_3_eigen"]:
    s_eigen = ElasticStructure(omega_init=0.0, tilt=0, cone=0, n_flexible_blades=3)
    freqs, modes, keep = s_eigen.eigen_analysis()
    dof_names = ["x_t"] + [f"q_b{b}_m{i}" for b in range(3) for i in range(3)]
    pd.DataFrame({"mode": np.arange(len(freqs)) + 1, "freq_Hz": freqs}).to_csv(
        dir_task_3 / "eigen_frequencies.csv", index=False
    )
    pd.DataFrame(modes, index=dof_names, columns=[f"mode_{i+1}" for i in range(len(freqs))]).to_csv(
        dir_task_3 / "eigen_mode_shapes.csv", index_label="DOF"
    )
    pd.DataFrame({"mode": np.arange(len(freqs)) + 1, "freq_Hz": freqs.round(3)}).to_csv(
        dir_task_3 / "eigen_frequencies_rounded.csv", index=False
    )
    pd.DataFrame(modes.round(3), index=dof_names, columns=[f"mode_{i+1}" for i in range(len(freqs))]).to_csv(
        dir_task_3 / "eigen_mode_shapes_rounded.csv", index_label="DOF"
    )


def run_q3(v0: float, turbulent: bool, case_dir: Path):
    case_dir.mkdir(exist_ok=True, parents=True)
    if turbulent:
        wind = make_turb(v0, T_q3, case_dir / "turb.nc")
    else:
        wind = WindWithTower(ShearWind(119, v0, 0.2))
    sim = Simulation(
        structure=ElasticStructure(omega_init=omega_init(v0), tilt=0, cone=0, n_flexible_blades=3),
        controller=PIController(),
        aerodynamics=Aerodynamics(),
        wind=wind,
        recorders=base_recorders(blade_idx=0, n_flexible_blades=3),
    )
    T = T_q3 if turbulent else T_q3_steady
    sim.run(dt_q3, T, case_dir, overwrite=True)


q3_cases = [
    (7.0, False, "v7_steady"),
    (7.0, True, "v7_turb"),
    (18.0, False, "v18_steady"),
    (18.0, True, "v18_turb"),
]

if do["task_3_sims"]:
    for v0, turbulent, name in q3_cases:
        run_q3(v0, turbulent, dir_task_3 / name)


if plot["task_3"]:
    if (dir_task_3 / "eigen_frequencies.csv").is_file():
        eigen = pd.read_csv(dir_task_3 / "eigen_frequencies.csv")
        fig, ax = plt.subplots()
        ax.stem(eigen["mode"], eigen["freq_Hz"])
        ax.set_xlabel("Mode #")
        ax.set_ylabel("Natural frequency (Hz)")
        ax.set_title("Q3 — 11-DOF natural frequencies (azimuth removed)")
        fig.tight_layout()
        fig.savefig(dir_task_3 / "eigen_frequencies.pdf")

    for _, _, name in q3_cases:
        case_dir = dir_task_3 / name
        if not (case_dir / "tip_deflections_0.csv").is_file():
            continue
        df_omega = pd.read_csv(case_dir / "rot_speed.csv")
        omega_mean = df_omega["omega"].iloc[len(df_omega) // 2 :].mean()

        # Tip deflections (3 blades)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for b in range(3):
            df = pd.read_csv(case_dir / f"tip_deflections_{b}.csv")
            axes[0].plot(df["time"], df["flapwise"], label=f"blade {b}")
            axes[1].plot(df["time"], df["edgewise"], label=f"blade {b}")
        axes[0].set_ylabel("Tip flap (m)")
        axes[1].set_ylabel("Tip edge (m)")
        axes[1].set_xlabel("Time (s)")
        axes[0].legend()
        fig.suptitle(f"Q3 — {name}")
        fig.tight_layout()
        fig.savefig(case_dir / "tip_deflections.pdf")

        # Tower + root moment
        df_tower = pd.read_csv(case_dir / "tower_position.csv")
        df_root = pd.read_csv(case_dir / "root_bending_blade_0.csv")
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(df_tower["time"], df_tower["x_t"])
        axes[0].set_ylabel("Tower disp. (m)")
        axes[1].plot(df_root["time"], df_root["flapwise"], label="flap")
        axes[1].plot(df_root["time"], df_root["edgewise"], label="edge")
        axes[1].set_ylabel("Root moment @ 2.8 m (N·m)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()
        fig.suptitle(f"Q3 — {name}")
        fig.tight_layout()
        fig.savefig(case_dir / "tower_root.pdf")

        # PSDs
        df_tip = pd.read_csv(case_dir / "tip_deflections_0.csv")
        signals = [
            (df_tip["flapwise"].to_numpy(), "Tip flap"),
            (df_tip["edgewise"].to_numpy(), "Tip edge"),
            (df_tower["x_t"].to_numpy(), "Tower disp."),
            (df_root["flapwise"].to_numpy(), "Root flap moment"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, (sig, label) in zip(axes.ravel(), signals):
            f, P = psd(sig, dt_q3)
            ax.semilogy(f, P)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(f"PSD of {label}")
            add_rotor_marks(ax, omega_mean)
        fig.suptitle(f"Q3 — PSDs, {name}")
        fig.tight_layout()
        fig.savefig(case_dir / "psd.pdf")

# ---------------------------------------------------------------------------
# Q3 — animate each system mode for 20 s
# ---------------------------------------------------------------------------

if plot["animate_system_modes"]:
    freqs_file = dir_task_3 / "eigen_frequencies.csv"
    modes_file = dir_task_3 / "eigen_mode_shapes.csv"
    animate_modes(dir_task_3 / "eigen_frequencies.csv", dir_task_3 / "eigen_mode_shapes.csv", dir_task_3 / "animations")
