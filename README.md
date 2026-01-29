# Stochastic-Technology-Diffusion-on-Networks

This folder contains all the code and data needed to reproduce the simulations
and figures in the report “Stochastic Technology Diffusion on Networks”.

To see the experiment results, please refer to "Stochastic Technology Diffusion on Networks.pdf" or "Stochastic_Technology_Diffusion_on_Networks__A_Network_SAR_Model.pdf".

----------------------------------------------------------------------
Directory structure
----------------------------------------------------------------------

code_and_csv_files/
│
├── README.txt
│
├── _src/
│   ├── experiments_main.py
│   ├── network_sar.py
│   ├── plot_psuccess_from_csv.py
│   ├── plot_peak_variance.py
│   ├── plot_single_trajectories.py
│   └── plot_so_trends.py
│
├── _results/
│   ├── overlay_betaSweep_peaks_g0.02_r0.005.csv
│   ├── overlay_betaSweep_peaks_g0.02_r0.005_critical_region.csv
│   ├── peak_variance_vs_beta_g0.02_r0.005.csv
│   └── StackOverflow-results/
│       ├── tag_monthly_counts.csv
│       └── total_monthly_counts.csv
│
└── _figures/        (generated automatically)

----------------------------------------------------------------------
Code overview (_src/)
----------------------------------------------------------------------

network_sar.py
  Core SAR (Susceptible–Active–Recovered) model implementation on networks
  and the Gillespie stochastic simulation algorithm.

experiments_main.py
  Main simulation driver. Runs all SAR simulations on ER and BA networks
  and outputs peak adoption data used throughout the report.

plot_psuccess_from_csv.py
  Computes and plots success probability P_success(β; θ) with binomial
  standard error bars (Report Section 3.4).

plot_peak_variance.py
  Computes and plots the variance of peak adoption versus β
  (Report Section 3.3).

plot_single_trajectories.py
  Plots a small number of individual trajectories to illustrate
  run-to-run variability (Report Section 3.5).

plot_so_trends.py
  Processes and plots Stack Overflow tag activity data
  (Report Section 4.2).

----------------------------------------------------------------------
Result files (_results/)
----------------------------------------------------------------------

overlay_betaSweep_peaks_g0.02_r0.005.csv
  Peak adoption values for a broad β sweep.

  Columns:
    - network   (ER or BA)
    - beta      (adoption rate)
    - run       (trajectory index)
    - peak      (A_max = max_t A(t)/N)

overlay_betaSweep_peaks_g0.02_r0.005_critical_region.csv
  Same format as above, but with a finer β grid focused on the
  critical transition region.

peak_variance_vs_beta_g0.02_r0.005.csv
  Variance of peak adoption as a function of β.

  Columns:
    - network
    - beta
    - var_peak

StackOverflow-results/
  tag_monthly_counts.csv
    Monthly question counts per technology tag.

  total_monthly_counts.csv
    Total number of Stack Overflow questions per month.

----------------------------------------------------------------------
Reproducing the results
----------------------------------------------------------------------

Run the scripts in the following order from the project root:

  python _src/experiments_main.py
  python _src/plot_psuccess_from_csv.py
  python _src/plot_peak_variance.py
  python _src/plot_single_trajectories.py
  python _src/plot_so_trends.py

All figures are saved automatically to the _figures/ directory.
