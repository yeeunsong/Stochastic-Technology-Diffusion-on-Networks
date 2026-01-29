# src/experiments.py
from __future__ import annotations

import os
import csv
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
    print(f"[saved] {os.path.abspath(path)}", flush=True)


def network_stats(name, G):
    N = G.number_of_nodes()
    E = G.number_of_edges()
    avg_k = 2 * E / N
    max_k = max(dict(G.degree()).values())
    return name, N, E, avg_k, max_k


def median_iqr(arr: np.ndarray):
    """arr shape (K,T) -> returns median,q25,q75 each shape (T,)"""
    med = np.median(arr, axis=0)
    q25 = np.quantile(arr, 0.25, axis=0)
    q75 = np.quantile(arr, 0.75, axis=0)
    return med, q25, q75


# =========================
# Core ensemble runner
# =========================

def run_ensemble_SAR(
    G,
    params: SARParams,
    N: int,
    months: int,
    K: int,
    A0: int,
    R0: int,
    seed0: int = 0,
):
    """
    Returns:
        t      : (T,) time grid (monthly, length months+1)
        S_all  : (K, T) S/N trajectories
        A_all  : (K, T) A/N trajectories
        R_all  : (K, T) R/N trajectories
    """
    S_all = np.zeros((K, months + 1), dtype=float)
    A_all = np.zeros((K, months + 1), dtype=float)
    R_all = np.zeros((K, months + 1), dtype=float)

    t = None
    for r in range(K):
        seed = seed0 + r
        t_r, S, A, R = gillespie_network_SAR(G, params, A0=A0, R0=R0, months=months, seed=seed)
        if t is None:
            t = t_r
        S_all[r] = S / N
        A_all[r] = A / N
        R_all[r] = R / N

    return t, S_all, A_all, R_all


# =========================
# Beta sweep experiment
# =========================

# Broad beta sweep values
# def beta_sweep_values():
#     return np.array([
#         0.0010,
#         0.0015,
#         0.0022,
#         0.0033,
#         0.0050,
#         0.0075,
#         0.0110,
#         0.0160,
#         0.0230,
#         0.0330,
#     ], dtype=float)


# Critical region beta sweep values
def beta_sweep_values():
    return np.array([
        0.0025,
        0.0031,
        0.0038,
        0.0047,
        0.0058,
    ], dtype=float)


def experiment_overlay_and_shared_bins_beta_sweep(
    fig_dir: str = "../_figures/Overlay_betaSweep",
    out_dir: str = "../_results",
    N: int = 10_000,
    months: int = 200,
    K: int = 200,
    A0: int = 10,
    R0: int = 0,
    gamma: float = 0.02,
    rho: float = 0.005,
    avg_k_target: float = 10.0,
    bins: int = 40,
    beta_values: np.ndarray | None = None,
):
    ensure_dir(fig_dir)
    ensure_dir(out_dir)

    fig_dir_abs = os.path.abspath(fig_dir)
    out_dir_abs = os.path.abspath(out_dir)
    print("Output fig dir:", fig_dir_abs, flush=True)
    print("Output csv dir:", out_dir_abs, flush=True)

    if beta_values is None:
        beta_values = beta_sweep_values()

    er_p = avg_k_target / (N - 1)
    G_er = make_network(N, "er", seed=1, er_p=er_p)

    ba_m = max(1, int(round(avg_k_target / 2)))  # BA avg degree ≈ 2m
    G_ba = make_network(N, "sf", seed=2, ba_m=ba_m)

    nets = [("ER", G_er, 1_000_000), ("BA", G_ba, 2_000_000)]

    print("\nNetwork sanity check:", flush=True)
    avg_k_map = {}
    for name, G, _ in nets:
        _, Nn, E, avg_k, max_k = network_stats(name, G)
        avg_k_map[name] = avg_k
        print(f"{name}: N={Nn}, avg_k={avg_k:.2f}, max_k={max_k}", flush=True)
    print("", flush=True)

    peaks_csv = os.path.join(out_dir, f"overlay_betaSweep_peaks_g{gamma}_r{rho}_critical_region.csv")
    quant_csv = os.path.join(out_dir, f"overlay_betaSweep_quantiles_g{gamma}_r{rho}_critical_region.csv")

    peaks_exists = os.path.exists(peaks_csv)
    quant_exists = os.path.exists(quant_csv)

    with open(peaks_csv, "a", newline="") as f_peaks, open(quant_csv, "a", newline="") as f_quant:
        peaks_writer = csv.DictWriter(f_peaks, fieldnames=["network", "beta", "run", "peak"])
        quant_writer = csv.DictWriter(f_quant, fieldnames=["network", "beta", "t", "A_med", "A_q25", "A_q75"])

        if not peaks_exists:
            peaks_writer.writeheader()
        if not quant_exists:
            quant_writer.writeheader()

        k_str = f"{avg_k_target:.1f}"
        gamma_str = f"{gamma:.4f}"
        rho_str = f"{rho:.4f}"

        # --- Sweep betas
        for bi, beta in enumerate(beta_values):
            beta_str = f"{beta:.4f}"
            print(f"\n=== beta {beta_str} ({bi+1}/{len(beta_values)}) ===", flush=True)

            params = SARParams(beta=float(beta), gamma=float(gamma), rho=float(rho))

            results = {}
            for name, G, net_seed_base in nets:
                seed0 = net_seed_base + 10_000 * bi  # new RNG stream per beta per network

                t, S_all, A_all, R_all = run_ensemble_SAR(
                    G, params, N=N, months=months, K=K, A0=A0, R0=R0, seed0=seed0
                )

                A_med, A_q25, A_q75 = median_iqr(A_all)
                peaks = A_all.max(axis=1)  # already A/N

                results[name] = dict(t=t, A_med=A_med, A_q25=A_q25, A_q75=A_q75, peaks=peaks)

                print(f"{name}: mean_peak={peaks.mean():.4f}  max_peak={peaks.max():.4f}", flush=True)

                # ---- write peaks rows (K rows)
                for r in range(K):
                    peaks_writer.writerow({
                        "network": name,
                        "beta": float(beta),
                        "run": int(r),
                        "peak": float(peaks[r]),
                    })
                f_peaks.flush()

                # ---- write quantile rows (T rows)
                for ti in range(len(t)):
                    quant_writer.writerow({
                        "network": name,
                        "beta": float(beta),
                        "t": float(t[ti]),
                        "A_med": float(A_med[ti]),
                        "A_q25": float(A_q25[ti]),
                        "A_q75": float(A_q75[ti]),
                    })
                f_quant.flush()

            # Overlay plot
            plt.figure(figsize=(7, 5))
            for name in ["ER", "BA"]:
                tt = results[name]["t"]
                plt.plot(tt, results[name]["A_med"], label=f"{name} median")
                plt.fill_between(tt, results[name]["A_q25"], results[name]["A_q75"], alpha=0.25, label=f"{name} IQR")

            plt.xlabel("Months")
            plt.ylabel("Active fraction A(t)/N")
            plt.title(
                f"A(t) median + IQR (K={K}), ER vs BA\n"
                f"β={beta_str}, γ={gamma_str}, ρ={rho_str}, ⟨k⟩≈{k_str}"
            )
            plt.legend()

            fname_overlay = (
                f"E_overlay_A_medianIQR_ERvsBA_"
                f"N{N}_m{months}_K{K}_k{k_str}_b{beta_str}_g{gamma_str}_r{rho_str}.png"
            )
            save_fig(os.path.join(fig_dir, fname_overlay))

            # Peak distribution histogram
            peaks_er = results["ER"]["peaks"]
            peaks_ba = results["BA"]["peaks"]

            xmin = float(min(peaks_er.min(), peaks_ba.min()))
            xmax = float(max(peaks_er.max(), peaks_ba.max()))
            pad = 0.02 * (xmax - xmin + 1e-12)
            xmin -= pad
            xmax += pad

            bin_edges = np.linspace(xmin, xmax, bins + 1)

            for name in ["ER", "BA"]:
                peaks = results[name]["peaks"]
                plt.figure(figsize=(6, 4))
                plt.hist(peaks, bins=bin_edges, alpha=0.85)
                plt.xlim(xmin, xmax)
                plt.xlabel("Peak adoption fraction  max_t A(t)/N")
                plt.ylabel("Count")
                plt.title(
                    f"{name}: Peak adoption distribution (K={K})\n"
                    f"β={beta_str}, γ={gamma_str}, ρ={rho_str}, ⟨k⟩≈{k_str}"
                )
                fname_hist = (
                    f"{name}_peakHist_sharedBins_"
                    f"N{N}_m{months}_K{K}_k{k_str}_b{beta_str}_g{gamma_str}_r{rho_str}.png"
                )
                save_fig(os.path.join(fig_dir, fname_hist))

            print(f"Done beta {beta_str}.", flush=True)

    print("\nAll betas done.", flush=True)
    print("Figures in:", fig_dir_abs, flush=True)
    print("CSV saved in:", out_dir_abs, flush=True)
    print("  -", os.path.abspath(peaks_csv), flush=True)
    print("  -", os.path.abspath(quant_csv), flush=True)


if __name__ == "__main__":
    experiment_overlay_and_shared_bins_beta_sweep(
        fig_dir="../_figures/Overlay_BetaSweep/betaSweep_g0.02_r0.005_critical_region",
        out_dir="../_results",
        N=10_000,
        months=200,
        K=200,
        A0=10,
        R0=0,
        gamma=0.02,
        rho=0.005,
        avg_k_target=10.0,
        bins=40,
        beta_values=beta_sweep_values(),
    )