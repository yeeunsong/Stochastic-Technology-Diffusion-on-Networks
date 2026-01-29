import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
CSV_IN = "../_results/overlay_betaSweep_peaks_g0.02_r0.005.csv"
OUT_DIR = "../_results"
FIG_DIR = "../_figures/Variance"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CSV_OUT = os.path.join(OUT_DIR, "peak_variance_vs_beta_g0.02_r0.005.csv")

# =========================
# Load data
# =========================
df = pd.read_csv(CSV_IN)

# Expected columns:
# network, beta, run, peak

# =========================
# Compute statistics
# =========================
stats = (
    df.groupby(["network", "beta"])["peak"]
      .agg(
          mean_peak="mean",
          var_peak=lambda x: np.var(x, ddof=1)  # unbiased variance
      )
      .reset_index()
)

# =========================
# Save CSV
# =========================
stats.to_csv(CSV_OUT, index=False)

print("\nVariance of peak adoption per beta:\n")
for _, row in stats.iterrows():
    print(
        f"{row['network']:>2} | "
        f"beta={row['beta']:.4f} | "
        f"mean_peak={row['mean_peak']:.4e} | "
        f"var={row['var_peak']:.4e}"
    )

print(f"\nSaved variance CSV to: {CSV_OUT}")

# =========================
# Plot
# =========================
plt.figure(figsize=(6, 4))

for net, sub in stats.groupby("network"):
    plt.plot(
        sub["beta"],
        sub["var_peak"],
        marker="o",
        label=net
    )

plt.xscale("log")
plt.xlabel(r"Adoption rate $\beta$")
plt.ylabel(r"$\mathrm{Var}(A_{\max})$")
plt.title("Variance of peak adoption vs adoption rate")
plt.legend()
plt.grid(alpha=0.3)

fig_path = os.path.join(FIG_DIR, "variance_peak_vs_beta.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"Saved variance plot to: {fig_path}")