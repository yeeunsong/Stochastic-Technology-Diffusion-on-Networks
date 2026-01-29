# _src/plot_psuccess_from_csv.py

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

CSV_PATH = "../_results/overlay_betaSweep_peaks_g0.02_r0.005.csv"
FIG_DIR = "../_figures/BetaSweep"

THETAS = [0.05, 0.08, 0.12]     # robust-theta set
K = 200                         # ensemble size
NETWORKS = ["ER", "BA"]


# =========================
# Utilities
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def binomial_stderr(p: float, k: int) -> float:
    """Binomial standard error."""
    return np.sqrt(p * (1.0 - p) / k)


# =========================
# Main plotting function
# =========================

def plot_psuccess_from_csv():
    ensure_dir(FIG_DIR)

    df = pd.read_csv(CSV_PATH)

    required_cols = {"network", "beta", "run", "peak"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required_cols}")

    beta_values = np.sort(df["beta"].unique())

    for net in NETWORKS:
        df_net = df[df["network"] == net]

        plt.figure(figsize=(6.5, 4.5))

        for theta in THETAS:
            P_vals = []
            err_vals = []

            for beta in beta_values:
                peaks = df_net[df_net["beta"] == beta]["peak"].values
                K_eff = len(peaks)

                # success probability
                P = np.mean(peaks >= theta)

                # binomial standard error
                err = binomial_stderr(P, K_eff)

                P_vals.append(P)
                err_vals.append(err)

            P_vals = np.array(P_vals)
            err_vals = np.array(err_vals)

            plt.errorbar(
                beta_values,
                P_vals,
                yerr=err_vals,
                marker="o",
                capsize=3,
                linewidth=1.5,
                label=rf"$\theta={theta:.2f}$",
            )

        # formatting
        plt.xscale("log")
        plt.ylim(-0.02, 1.02)
        plt.xlabel(r"Adoption rate $\beta$")
        plt.ylabel(r"$P_{\mathrm{success}}$")
        plt.title(f"{net}: Success probability vs $\\beta$\n(binomial error bars)")
        plt.legend()
        plt.grid(alpha=0.3, which="both")

        out_path = os.path.join(
            FIG_DIR,
            f"{net}_Psuccess_beta_robust_theta_g0.02_r0.005_v3.png"
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved: {out_path}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    plot_psuccess_from_csv()