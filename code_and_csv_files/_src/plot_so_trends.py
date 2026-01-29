import pandas as pd

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


TAG_COUNTS_CSV = "StackOverflow-results/tag_monthly_counts.csv"      # from Query 1
TOTAL_COUNTS_CSV = "StackOverflow-results/total_monthly_counts.csv"  # from Query 2

# Load data
tags = pd.read_csv(TAG_COUNTS_CSV)
total = pd.read_csv(TOTAL_COUNTS_CSV)

# Expected columns:
# tags:  year, month, tag, question_count
# total: year, month, total_questions

tags["date"] = pd.to_datetime(tags["year"].astype(str) + "-" + tags["month"].astype(str) + "-01")
total["date"] = pd.to_datetime(total["year"].astype(str) + "-" + total["month"].astype(str) + "-01")

df = tags.merge(total[["year", "month", "total_questions", "date"]],
                on=["year", "month", "date"],
                how="left")

if df["total_questions"].isna().any():
    raise ValueError("Some rows did not match total_questions. Check your CSV columns / merge keys.")

df["norm_activity"] = df["question_count"] / df["total_questions"]

wide_counts = df.pivot_table(index="date", columns="tag", values="question_count", aggfunc="sum").sort_index()
wide_norm = df.pivot_table(index="date", columns="tag", values="norm_activity", aggfunc="sum").sort_index()

# smoothing to reduce month-to-month noise
WINDOW = 6  # months
wide_norm_smooth = wide_norm.rolling(WINDOW, min_periods=1).mean()

# Plot raw counts
plt.figure()
wide_counts.plot(ax=plt.gca())
plt.title("Stack Overflow tag questions per month (raw counts)")
plt.xlabel("Date")
plt.ylabel("Questions / month")
plt.tight_layout()
plt.savefig("StackOverflow-plots/01_raw_counts.png", dpi=200)
plt.close()

# Plot normalized activity
plt.figure()
wide_norm.plot(ax=plt.gca())
plt.title("Stack Overflow tag activity (normalized by total questions)")
plt.xlabel("Date")
plt.ylabel("Share of all SO questions")
plt.tight_layout()
plt.savefig("StackOverflow-plots/02_normalized.png", dpi=200)
plt.close()

# Plot smoothed normalized activity
plt.figure()
wide_norm_smooth.plot(ax=plt.gca())
plt.title(f"Normalized tag activity (smoothed: {WINDOW}-month rolling mean)")
plt.xlabel("Date")
plt.ylabel("Share of all SO questions")
plt.tight_layout()
plt.savefig("StackOverflow-plots/03_normalized_smoothed.png", dpi=200)
plt.close()

print("\nPeak normalized activity (share) by tag:")
print(wide_norm.max().sort_values(ascending=False))

print("\nMost recent normalized activity (share) by tag:")
print(wide_norm.dropna(how="all").iloc[-1].sort_values(ascending=False))