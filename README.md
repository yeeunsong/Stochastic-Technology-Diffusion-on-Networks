# Stochastic Technology Diffusion on Networks

This repository contains all code and data required to reproduce the simulations
and figures presented in the report **“Stochastic Technology Diffusion on Networks.”**

To see the experiment results, please refer to **"Stochastic Technology Diffusion on Networks.pdf"** or **"Stochastic_Technology_Diffusion_on_Networks__A_Network_SAR_Model.pdf"**.


---

## Code Overview (`_src/`)

#### `network_sar.py`
Core implementation of the SAR (Susceptible–Active–Recovered) model on networks,
including the Gillespie stochastic simulation algorithm.

#### `experiments_main.py`
Main simulation driver. Runs SAR simulations on Erdős–Rényi (ER) and
Barabási–Albert (BA) networks and outputs peak adoption data used throughout
the report.

#### `plot_psuccess_from_csv.py`
Computes and plots the success probability  with binomial standard error bars.

#### `plot_peak_variance.py`
Computes and plots the variance of peak adoption as a function of the
adoption rate beta.

#### `plot_single_trajectories.py`
Plots a small number of individual trajectories to illustrate run-to-run
stochastic variability.

#### `plot_so_trends.py`
Processes and plots Stack Overflow tag activity data used for qualitative
comparison with the model.

---

## Result Files (`_results/`)

#### `overlay_betaSweep_peaks_g0.02_r0.005.csv`
Peak adoption values from a broad sweep over the adoption rate \( \beta \).

**Columns**
- `network` — Network type (`ER` or `BA`)
- `beta` — Adoption rate
- `run` — Trajectory index
- `peak` — Peak adoption  

#### `overlay_betaSweep_peaks_g0.02_r0.005_critical_region.csv`
Same format as above, but with a finer beta grid focused on the
critical transition region.

#### `peak_variance_vs_beta_g0.02_r0.005.csv`
Variance of peak adoption as a function of beta.

**Columns**
- `network`
- `beta`
- `var_peak`

#### `StackOverflow-results/`

- `tag_monthly_counts.csv`  
  Monthly Stack Overflow question counts per technology tag.

- `total_monthly_counts.csv`  
  Total number of Stack Overflow questions per month.

---

## Reproducing the Results

Run the following scripts from the project root directory, in order:

```bash
python _src/experiments_main.py
python _src/plot_psuccess_from_csv.py
python _src/plot_peak_variance.py
python _src/plot_single_trajectories.py
python _src/plot_so_trends.py
```
All figures are saved automatically to the _figures/ directory.
