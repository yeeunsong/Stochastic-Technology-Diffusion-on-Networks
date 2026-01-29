from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from network_sar import SARParams, make_network, gillespie_network_SAR


# =========================
# Utilities
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(path: str, dpi: int = 200):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


# =========================
# Single-trajectory experiment
# =========================

def plot_single_trajectories_beta_sweep(
    fig_dir: str = "../_figures/SingleTrajectories",
    network_type: str = "ER",          # "ER" or "BA"
    N: int = 10_000,
    months: int = 200,
    K_traj: int = 10,                  # number of trajectories per beta
    A0: int = 10,
    R0: int = 0,
    gamma: float = 0.02,
    rho: float = 0.005,
    avg_k_target: float = 10.0,
    beta_values: np.ndarray | None = None,
):
    """
    Plot K_traj individual stochastic trajectories A(t)/N
    for each beta value.
    """

    ensure_dir(fig_dir)

    if beta_values is None:
        beta_values = np.array([
            0.0010, 0.0015, 0.0022, 0.0033,
            0.0050, 0.0075, 0.0110, 0.0160,
            0.0230, 0.0330
        ])

    if network_type == "ER":
        er_p = avg_k_target / (N - 1)
        G = make_network(N, "er", seed=1, er_p=er_p)
    elif network_type == "BA":
        m = max(1, int(round(avg_k_target / 2)))
        G = make_network(N, "sf", seed=2, ba_m=m)
    else:
        raise ValueError("network_type must be 'ER' or 'BA'")

    print(f"\nSingle-trajectory plots for {network_type}")
    print(f"N={N}, months={months}, K_traj={K_traj}")

    # ---- beta sweep
    for bi, beta in enumerate(beta_values):
        params = SARParams(beta=float(beta), gamma=gamma, rho=rho)

        plt.figure(figsize=(7, 5))

        for r in range(K_traj):
            seed = 100_000 * bi + r
            t, S, A, R = gillespie_network_SAR(
                G,
                params,
                A0=A0,
                R0=R0,
                months=months,
                seed=seed,
            )

            plt.plot(t, A / N, lw=1.0, alpha=0.8)

        plt.xlabel("Months")
        plt.ylabel("Active fraction A(t)/N")
        plt.title(
            f"{network_type}: Example trajectories (K={K_traj})\n"
            f"β={beta:.4f}, γ={gamma}, ρ={rho}, ⟨k⟩≈{avg_k_target}"
        )

        fname = (
            f"{network_type}_singleTraj_"
            f"N{N}_m{months}_K{K_traj}_"
            f"beta{beta:.4f}_gamma{gamma:.3f}_rho{rho:.3f}.png"
        )
        save_fig(os.path.join(fig_dir, fname))

        print(f"Saved: {fname}")


# =========================
# Run
# =========================

if __name__ == "__main__":
    # braod beta sweep values
    # beta_values = np.array([
    #     0.0010, 0.0015, 0.0022, 0.0033,
    #     0.0050, 0.0075, 0.0110, 0.0160,
    #     0.0230, 0.0330
    # ])

    # critical region beta sweep values
    beta_critical = np.array([
        0.0025,
        0.0028,
        0.0031,
        0.0034,
        0.0038,
        0.0042,
        0.0047,
        0.0052,
        0.0058,
        0.0065,
    ])

    # ER
    plot_single_trajectories_beta_sweep(
        fig_dir="../_figures/SingleTrajectories/g0.02_r0.005_critical_region/ER",
        network_type="ER",
        beta_values=beta_critical,
    )

    # BA
    plot_single_trajectories_beta_sweep(
        fig_dir="../_figures/SingleTrajectories/g0.02_r0.005_critical_region/BA",
        network_type="BA",
        beta_values=beta_critical,
    )