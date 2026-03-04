import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Parameters ---
A = 0.2  # m, vibration amplitude
omega = 3.0  # rad/s, vibration frequency
V0 = 5.0  # m/s, inflow speed
c = 1  # m, chord length
rho = 1.225  # kg/m³, air density
N_cycles = 5  # total simulation cycles (first skipped in integration)

# alpha0_list = np.array([0])  # degrees
alpha0_list = np.array([0, 5, 10, 15, 20])  # degrees
theta_list = np.arange(0, 361)  # degrees, 0..360
theta_plot = 90  # degrees; theta for which Cl/Cd loops over the polar are plotted

# Switch: which model results to use in figs 3 & 4 (True = dynamic stall, False = quasi-steady)
figs34_dynamic_stall = False

# --- Load FFA 241 polar (rel_thickness = 24.1%) ---
polar = pd.read_csv("rel_t_241.csv")
alpha_tab = np.deg2rad(polar["alpha"].to_numpy())
cl_tab = polar["cl_stdy"].to_numpy()
cd_tab = polar["cd_stdy"].to_numpy()
fs_tab = polar["f_s"].to_numpy()
clinv_tab = polar["cl_inv"].to_numpy()
clfs_tab = polar["cl_fs"].to_numpy()

# --- Time discretisation ---
T_period = 2 * np.pi / omega  # one oscillation period [s]
dt = 0.05  # time step [s]
t = np.arange(0, N_cycles * T_period + dt / 2, dt)
t_last = (N_cycles - 1) * T_period
t_int = t[t >= t_last]

# Angle arrays shaped for broadcasting over (n_alpha0, n_theta)
alpha0_rad = np.deg2rad(alpha0_list)[:, None]  # (n_alpha0, 1)
theta_rad = np.deg2rad(theta_list)[None, :]  # (1, n_theta)


def compute_work(dynamic_stall=False):
    """
    Compute aerodynamic work W = A*omega * integral{ F_x(t) cos(omega*t) dt }
    over the last full oscillation period, sweeping all alpha_0 and theta combinations.

    Section force in vibration direction: F_x = q*c*(C_l*sin(alpha-theta) - C_d*cos(alpha-theta))
    Aerodynamic work: W = A*omega * integral{ F_x(t)*cos(omega*t) dt }

    When dynamic_stall=True a first-order lag (Øye model) is applied to the
    separation state f_s; otherwise quasi-steady coefficients are used directly.

    Parameters
    ----------
    dynamic_stall : bool
        If True, apply dynamic stall model. If False, use quasi-steady coefficients.

    Returns
    -------
    W : ndarray, shape (n_alpha0, n_theta)
        Aerodynamic work per unit span over the last oscillation period.
    alpha_loop : ndarray, shape (n_int, n_alpha0)
        Angle of attack time series over the last period at theta = theta_plot.
    cl_loop : ndarray, shape (n_int, n_alpha0)
        Lift coefficient time series over the last period at theta = theta_plot.
    cd_loop : ndarray, shape (n_int, n_alpha0)
        Drag coefficient time series over the last period at theta = theta_plot.
    vrel_loop : ndarray, shape (n_int, n_alpha0)
        Relative velocity magnitude time series over the last period at theta = theta_plot.
    """
    # Separation state initialised to fully attached (f_s = 1); only used when dynamic_stall=True
    fs_state = np.ones((len(alpha0_list), len(theta_list)))

    # Pre-allocate arrays for the last period
    n_int = len(t_int)
    Fx_arr = np.zeros((n_int, len(alpha0_list), len(theta_list)))
    alpha_loop = np.zeros((n_int, len(alpha0_list)))  # (n_int, n_alpha0) for theta_plot
    cl_loop = np.zeros((n_int, len(alpha0_list)))
    cd_loop = np.zeros((n_int, len(alpha0_list)))
    vrel_loop = np.zeros((n_int, len(alpha0_list)))
    i_theta_plot = np.argmin(np.abs(theta_list - theta_plot))
    i_int = 0

    for ti in t:
        xdot = A * omega * np.cos(omega * ti)  # airfoil velocity [m/s]

        # Relative velocity components in y-z plane
        vy = V0 * np.cos(alpha0_rad) + xdot * np.cos(theta_rad)  # (n_alpha0, n_theta)
        vz = V0 * np.sin(alpha0_rad) + xdot * np.sin(theta_rad)
        vrel = np.sqrt(vy**2 + vz**2)
        alpha = np.arctan2(vz, vy)

        # Quasi-steady aerodynamic coefficients
        cl_qs = np.interp(alpha, alpha_tab, cl_tab)
        cd = np.interp(alpha, alpha_tab, cd_tab)

        if dynamic_stall:
            # First-order lag on separation state (Øye model)
            fs_qs = np.interp(alpha, alpha_tab, fs_tab)
            clinv = np.interp(alpha, alpha_tab, clinv_tab)
            clfs_val = np.interp(alpha, alpha_tab, clfs_tab)
            tau = 4 * c / vrel
            fs_state = fs_qs + (fs_state - fs_qs) * np.exp(-dt / tau)
            cl = clinv * fs_state + clfs_val * (1 - fs_state)
        else:
            cl = cl_qs

        if ti >= t_last:
            # Section force in vibration direction x
            q = 0.5 * rho * vrel**2
            Fx = q * c * (cl * np.sin(alpha - theta_rad) - cd * np.cos(alpha - theta_rad))
            Fx_arr[i_int] = Fx
            alpha_loop[i_int] = alpha[:, i_theta_plot]
            cl_loop[i_int] = cl[:, i_theta_plot]
            cd_loop[i_int] = cd[:, i_theta_plot]
            vrel_loop[i_int] = vrel[:, i_theta_plot]
            i_int += 1

    # Integrate W = A*omega * integral{ F_x(t) cos(omega*t) dt } over the last period
    integrand = Fx_arr * np.cos(omega * t_int)[:, None, None]  # (n_int, n_alpha0, n_theta)
    W = np.asarray(A * omega * np.trapezoid(integrand, t_int, axis=0))  # (n_alpha0, n_theta)

    return W, alpha_loop, cl_loop, cd_loop, vrel_loop


# --- Run both cases ---
print("Computing quasi-steady.")
W_qs, alpha_loop_qs, cl_loop_qs, cd_loop_qs, vrel_loop_qs = compute_work(dynamic_stall=False)
print("Computing with dynamic stall.")
W_ds, alpha_loop_ds, cl_loop_ds, cd_loop_ds, vrel_loop_ds = compute_work(dynamic_stall=True)

# --- Select data for figs 3 & 4 based on switch ---
if figs34_dynamic_stall:
    alpha_loop_34 = alpha_loop_ds
    cl_loop_34 = cl_loop_ds
    cd_loop_34 = cd_loop_ds
    vrel_loop_34 = vrel_loop_ds
    figs34_label = "dynamic stall"
else:
    alpha_loop_34 = alpha_loop_qs
    cl_loop_34 = cl_loop_qs
    cd_loop_34 = cd_loop_qs
    vrel_loop_34 = vrel_loop_qs
    figs34_label = "quasi-steady"

# --- Fig 1: Aerodynamic work W vs vibration direction theta ---
print("Start plotting.")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

colors = [f"C0{i}" for i in range(len(alpha0_list))]
for idx, (a0, color) in enumerate(zip(alpha0_list, colors)):
    ax1.plot(theta_list, W_qs[idx], color=color, label=rf"$\alpha_0 = {a0}°$")
    ax2.plot(theta_list, W_ds[idx], color=color, label=rf"$\alpha_0 = {a0}°$")

for ax, title in zip((ax1, ax2), ("Quasi-steady", "Dynamic stall")):
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.axvline(theta_plot, color="k", linewidth=0.8, linestyle=":", label=rf"$\theta = {theta_plot}°$")
    ax.set_xlim(0, 360)
    ax.set_ylabel("Aerodynamic work $W$ of the last period (J/m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

ax2.set_xlabel(r"Vibration direction $\theta$ (°)")
fig.suptitle(rf"Exercise 3 — FFA 241, $A = {A}$ m, $\omega = {omega}$ rad/s, $V_0 = {V0}$ m/s, $c = {c}$ m")
plt.tight_layout()
plt.savefig("exercise_3_work.pdf")

# --- Fig 2: C_l polar loops for theta = theta_plot (QS and dynamic stall) ---
alpha_tab_deg = np.rad2deg(alpha_tab)

# x-axis range: quasi-steady alpha extremes ± 10 degrees
qs_alpha_deg = np.rad2deg(alpha_loop_qs)
xlim = (qs_alpha_deg.min() - 10, qs_alpha_deg.max() + 10)

mid = len(t_int) // 4  # index used to place direction arrows

fig2, (ax_cl1, ax_cl2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
for ax_cl, (label, alpha_loop, cl_loop), arrows in zip(
    (ax_cl1, ax_cl2),
    [("Quasi-steady", alpha_loop_qs, cl_loop_qs), ("Dynamic stall", alpha_loop_ds, cl_loop_ds)],
    [False, True],
):
    ax_cl.plot(alpha_tab_deg, cl_tab, "k--", linewidth=0.8, label="Static polar")

    for ia, (a0, color) in enumerate(zip(alpha0_list, colors)):
        alpha_deg = np.rad2deg(alpha_loop[:, ia])
        # Mark the start of the loop period; label only once to avoid duplicate legend entries
        ax_cl.plot(
            alpha_deg[0],
            cl_loop[0, ia],
            "ko",
            markersize=4,
            label="Loop start" if ia == 0 else "_nolegend_",
        )
        ax_cl.plot(alpha_deg, cl_loop[:, ia], color=color, label=rf"$\alpha_0 = {a0}°$")

        if arrows:
            ax_cl.annotate(
                "",
                xy=(np.rad2deg(alpha_loop[mid + 1, ia]), cl_loop[mid + 1, ia]),
                xytext=(np.rad2deg(alpha_loop[mid, ia]), cl_loop[mid, ia]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            )

    ax_cl.set_xlim(xlim)
    ax_cl.set_title(label)
    ax_cl.set_ylabel("$C_l$ (-)")
    ax_cl.grid(True, alpha=0.3)
    ax_cl.legend(fontsize=8)

ax_cl2.set_xlabel(r"$\alpha$ (°)")

fig2.suptitle(rf"Polar loops at $\theta = {theta_plot}°$ — $A = {A}$ m, $\omega = {omega}$ rad/s, $V_0 = {V0}$ m/s")
plt.tight_layout()
plt.savefig("exercise_3_polar_loops.pdf")

# --- Fig 3: Force component time series over the last period ---
# Uses data selected by figs34_dynamic_stall switch
theta_plot_rad = np.deg2rad(theta_plot)
t_period = t_int - t_last  # time within last period

q_34 = 0.5 * rho * vrel_loop_34**2 * c  # dynamic pressure * chord  (n_int, n_alpha0)
lift_34 = q_34 * cl_loop_34
drag_34 = q_34 * cd_loop_34
sin_term = np.sin(alpha_loop_34 - theta_plot_rad)
cos_term = np.cos(alpha_loop_34 - theta_plot_rad)

quantities = [
    (sin_term, r"$\sin(\alpha - \theta)$ (-)"),
    (-cos_term, r"$-\cos(\alpha - \theta)$ (-)"),
    (lift_34, "Lift (N/m)"),
    (drag_34, "Drag (N/m)"),
    (sin_term * lift_34, r"$\sin(\alpha - \theta) \cdot$ Lift (N/m)"),
    (-cos_term * drag_34, r"$-\cos(\alpha - \theta) \cdot$ Drag (N/m)"),
]

fig3, axes3 = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
for ax, (data, ylabel) in zip(axes3.flat, quantities):
    for ia, (a0, color) in enumerate(zip(alpha0_list, colors)):
        ax.plot(t_period, data[:, ia], color=color, label=rf"$\alpha_0 = {a0}°$")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
# Show tick labels and individual x labels on each bottom subplot (one per column)
for ax in axes3[-1]:
    ax.set_xlabel("Time in last period (s)")

fig3.suptitle(
    rf"Force components at $\theta = {theta_plot}°$ ({figs34_label}) — $A = {A}$ m, $\omega = {omega}$ rad/s, $V_0 = {V0}$ m/s"
)
fig3.tight_layout()
plt.savefig("exercise_3_components.pdf")

# --- Fig 4: Work integrand decomposition ---
# Uses data selected by figs34_dynamic_stall switch
from matplotlib.gridspec import GridSpec

cos_omega_t = np.cos(omega * t_int)[:, None]  # (n_int, 1) for broadcasting

fig4 = plt.figure(figsize=(10, 10))
gs4 = GridSpec(3, 2, fig4, hspace=0.45, wspace=0.35)
ax4_top_l = fig4.add_subplot(gs4[0, 0])
ax4_top_r = fig4.add_subplot(gs4[0, 1], sharex=ax4_top_l)
ax4_mid_l = fig4.add_subplot(gs4[1, 0], sharex=ax4_top_l)
ax4_mid_r = fig4.add_subplot(gs4[1, 1], sharex=ax4_top_l)
ax4_bot = fig4.add_subplot(gs4[2, :], sharex=ax4_top_l)
# Hide x tick labels for the top row; mid and bot rows show them
for ax in (ax4_top_l, ax4_top_r):
    plt.setp(ax.get_xticklabels(), visible=False)

# Top left: sin(α-θ)·L on left axis, cos(ωt) on right axis
ax4_top_l_r = ax4_top_l.twinx()
for ia, (a0, color) in enumerate(zip(alpha0_list, colors)):
    ax4_top_l.plot(t_period, (sin_term * lift_34)[:, ia], color=color, label=rf"$\alpha_0 = {a0}°$")
ax4_top_l_r.plot(t_period, np.cos(omega * t_int), "k:", linewidth=1.2, label=r"$\cos(\omega t)$")
ax4_top_l.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax4_top_l.set_ylabel(r"$\sin(\alpha-\theta)\cdot L$ (N/m)")
ax4_top_l_r.set_ylabel(r"$\cos(\omega t)$ (-)")
ax4_top_l.grid(True, alpha=0.3)
lines_l, labels_l = ax4_top_l.get_legend_handles_labels()
lines_r, labels_r = ax4_top_l_r.get_legend_handles_labels()
ax4_top_l.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7)

# Top right: -cos(α-θ)·D on left axis, cos(ωt) on right axis
ax4_top_r_r = ax4_top_r.twinx()
for ia, (a0, color) in enumerate(zip(alpha0_list, colors)):
    ax4_top_r.plot(t_period, (-cos_term * drag_34)[:, ia], color=color, label=rf"$\alpha_0 = {a0}°$")
ax4_top_r_r.plot(t_period, np.cos(omega * t_int), "k:", linewidth=1.2, label=r"$\cos(\omega t)$")
ax4_top_r.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax4_top_r.set_ylabel(r"$-\cos(\alpha-\theta)\cdot D$ (N/m)")
ax4_top_r_r.set_ylabel(r"$\cos(\omega t)$ (-)")
ax4_top_r.grid(True, alpha=0.3)
lines_l, labels_l = ax4_top_r.get_legend_handles_labels()
lines_r, labels_r = ax4_top_r_r.get_legend_handles_labels()
ax4_top_r.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7)

# Align y=0 for left and right axes in each top subplot
for ax_l, ax_r in [(ax4_top_l, ax4_top_l_r), (ax4_top_r, ax4_top_r_r)]:
    for ax in (ax_l, ax_r):
        lim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
        ax.set_ylim(-lim, lim)

# Middle left: sin(α-θ)·L·cos(ωt)
for ia, (a0, color) in enumerate(zip(alpha0_list, colors)):
    ax4_mid_l.plot(t_period, (sin_term * lift_34 * cos_omega_t)[:, ia], color=color, label=rf"$\alpha_0 = {a0}°$")
ax4_mid_l.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax4_mid_l.set_ylabel(r"$\sin(\alpha-\theta)\cdot L\cdot\cos(\omega t)$ (N/m)")
ax4_mid_l.set_xlabel("Time in last period (s)")
ax4_mid_l.grid(True, alpha=0.3)
ax4_mid_l.legend(fontsize=7)

# Middle right: -cos(α-θ)·D·cos(ωt)
for ia, (a0, color) in enumerate(zip(alpha0_list, colors)):
    ax4_mid_r.plot(t_period, (-cos_term * drag_34 * cos_omega_t)[:, ia], color=color, label=rf"$\alpha_0 = {a0}°$")
ax4_mid_r.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax4_mid_r.set_ylabel(r"$-\cos(\alpha-\theta)\cdot D\cdot\cos(\omega t)$ (N/m)")
ax4_mid_r.set_xlabel("Time in last period (s)")
ax4_mid_r.grid(True, alpha=0.3)
ax4_mid_r.legend(fontsize=7)

# Bottom row: sum = F_x * cos(omega*t), the work integrand
for ia, (a0, color) in enumerate(zip(alpha0_list, colors)):
    ax4_bot.plot(
        t_period,
        ((sin_term * lift_34 - cos_term * drag_34) * cos_omega_t)[:, ia],
        color=color,
        label=rf"$\alpha_0 = {a0}°$",
    )
ax4_bot.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax4_bot.set_xlabel("Time in last period (s)")
ax4_bot.set_ylabel("$F_x\cos(\omega t)$ (N/m)")
ax4_bot.grid(True, alpha=0.3)
ax4_bot.legend(fontsize=7)

fig4.suptitle(
    rf"Work integrand decomposition at $\theta = {theta_plot}°$ ({figs34_label}) — $A = {A}$ m, $\omega = {omega}$ rad/s, $V_0 = {V0}$ m/s"
)
fig4.tight_layout()
plt.savefig("exercise_3_work_integrand.pdf")
plt.show()
