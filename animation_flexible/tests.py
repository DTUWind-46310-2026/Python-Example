from animate_structure import animate
import numpy as np
from pathlib import Path
import pandas as pd


def test_5dof():
    state_path = Path("data/5dof/flexible_displacements.csv")
    pitch_path = Path("data/5dof/pitch.csv")
    output_path = Path("data/5dof/5dof.mp4")

    animate(
        state_csv=state_path,
        output=output_path,
        pitch_csv=pitch_path,
        T=(0, 60),
        fps=20,
    )


def test_11dof():
    state_path = Path("data/11dof/flexible_displacements.csv")
    pitch_path = Path("data/11dof/pitch.csv")
    output_path = Path("data/11dof/11dof.mp4")

    animate(
        state_csv=state_path,
        output=output_path,
        pitch_csv=pitch_path,
        n_tip_deflections=40,
        T=(0, 60),
        fps=20,
    )


def test_helikopter():
    t = np.linspace(0, 20, 600)
    omega = np.concat((np.linspace(0, 8, 450), 8 * np.ones(150)))
    q_0 = np.concat((-15 * (np.tanh(np.linspace(-8, 3, 450)) - 0.7), -15 * (np.tanh(3) - 0.7) * np.ones(150)))
    n_blades = 6
    pd.DataFrame(
        {"time": t, "x_t": np.zeros_like(t), "phi_shaft": np.cumsum(omega * t[1] * np.ones_like(t))}
        | {f"q_b{b}_m0": q_0 for b in range(n_blades)}
        | {f"q_b{b}_m{m}": np.zeros_like(t) for b in range(n_blades) for m in range(1, 3)}
    ).to_csv("data/helikopterhelikopter/flexible_displacements.csv", index=False)

    state_path = Path("data/helikopterhelikopter/flexible_displacements.csv")
    output_path = Path("data/helikopterhelikopter/helikopterhelikopter.mp4")

    animate(
        state_csv=state_path,
        output=output_path,
        tip_deflection=False,
        title="HELIKOPTER HELIKOPTER",
        tilt=-90,
        minmax_deflections=(-6, 30),
    )


def test_pitching():
    t = np.linspace(0, 10, 101)
    pd.DataFrame(
        {"time": t, "x_t": np.zeros_like(t), "phi_shaft": np.zeros_like(t)}
        | {f"q_b{b}_m{m}": np.zeros_like(t) for b in range(3) for m in range(3)}
    ).to_csv("data/pitching/flexible_displacements.csv", index=False)

    pd.DataFrame({"time": t, "pitch": np.rad2deg(np.linspace(0, 4 * np.pi, t.size))}).to_csv(
        "data/pitching/pitch.csv", index=False
    )

    state_path = Path("data/pitching/flexible_displacements.csv")
    pitch_path = Path("data/pitching/pitch.csv")
    output_path = Path("data/pitching/pitching.mp4")

    animate(
        state_csv=state_path,
        output=output_path,
        pitch_csv=pitch_path,
        tip_deflection=False,
        plot_blade=None,
        fps=10,
        title="Pitching only",
    )


def test_tower():
    t = np.linspace(0, 10, 101)
    pd.DataFrame(
        {"time": t, "x_t": 20 * np.sin(np.linspace(0, 2 * np.pi, t.size)), "phi_shaft": np.zeros_like(t)}
        | {f"q_b{b}_m{m}": np.zeros_like(t) for b in range(3) for m in range(3)}
    ).to_csv("data/tower/flexible_displacements.csv", index=False)

    state_path = Path("data/tower/flexible_displacements.csv")
    output_path = Path("data/tower/tower.mp4")

    animate(
        state_csv=state_path,
        output=output_path,
        tip_deflection=False,
        plot_blade=None,
        fps=10,
        title="Tower only",
    )


def test_mode_animation():
    from animate_structure import animate_modes

    animate_modes(
        "data/mode_animation/eigen_frequencies.csv", "data/mode_animation/eigen_mode_shapes.csv", "data/mode_animation/"
    )


if __name__ == "__main__":
    test_5dof()
    test_11dof()
    test_helikopter()
    test_pitching()
    test_tower()
    test_mode_animation()
