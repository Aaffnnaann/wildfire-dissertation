"""
Exploratory data analysis of the Mesogeos Track A (fire danger forecasting) dataset.

Reads positives.csv / negatives.csv (long format: one sample = 30 daily rows,
time_idx 0..29, day 29 = last observed day before the target day t) and produces
the figures for the Data chapter into ../figures/.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path(r"C:\Users\Afnan\Desktop\Dissertation\data")
FIGS = Path(r"C:\Users\Afnan\Desktop\Dissertation\figures")
FIGS.mkdir(exist_ok=True)

# ---- palette (validated reference palette, light mode) ----
FIRE = "#e34948"      # categorical red  -> fire class
NOFIRE = "#2a78d6"    # categorical blue -> no-fire class
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

plt.rcParams.update({
    "font.family": "Segoe UI",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": BASE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "figure.dpi": 200,
})

def style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)

print("loading csvs...")
pos = pd.read_csv(DATA / "positives.csv", parse_dates=["time"])
neg = pd.read_csv(DATA / "negatives.csv", parse_dates=["time"])
pos["label"], neg["label"] = 1, 0

n_pos, n_neg = pos["sample"].nunique(), neg["sample"].nunique()
print(f"samples: fire={n_pos}, no-fire={n_neg}")

# last observed day (closest to the fire/target day)
pos_last = pos[pos.time_idx == 29].copy()
neg_last = neg[neg.time_idx == 29].copy()

# unit conversions for readability
for df in (pos, neg, pos_last, neg_last):
    df["t2m_c"] = df["t2m"] - 273.15          # K -> deg C
    df["lst_day_c"] = df["lst_day"] - 273.15
    df["rh_pct"] = df["rh"] * 100             # fraction -> %
    df["tp_mm"] = df["tp"] * 1000             # m -> mm

# ---------------------------------------------------------------- fig 1: class balance + seasonality
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

ax = axes[0]
bars = ax.bar(["Fire", "No fire"], [n_pos, n_neg], color=[FIRE, NOFIRE], width=0.55)
for b, v in zip(bars, [n_pos, n_neg]):
    ax.text(b.get_x() + b.get_width() / 2, v + 250, f"{v:,}", ha="center",
            fontsize=11, fontweight="bold", color=INK)
ax.set_title("Class balance (samples)")
ax.set_ylim(0, n_neg * 1.15)
style_ax(ax)

ax = axes[1]
years = pos_last["time"].dt.year.value_counts().sort_index()
ax.bar(years.index, years.values, color=FIRE, width=0.7)
ax.set_title("Fire samples per year")
ax.set_xticks(range(2006, 2023, 4))
style_ax(ax)

ax = axes[2]
months = pos_last["time"].dt.month.value_counts().sort_index()
ax.bar(months.index, months.values, color=FIRE, width=0.7)
ax.set_title("Fire samples per month")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
style_ax(ax)

fig.tight_layout()
fig.savefig(FIGS / "fig1_class_balance_seasonality.png", bbox_inches="tight")
plt.close(fig)
print("fig1 done")

# ---------------------------------------------------------------- fig 2: spatial distribution
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.scatter(neg_last["x"], neg_last["y"], s=2, c=NOFIRE, alpha=0.25, linewidths=0, label="No fire")
ax.scatter(pos_last["x"], pos_last["y"], s=2, c=FIRE, alpha=0.45, linewidths=0, label="Fire ignition")
ax.set_title("Sample locations across the Mediterranean (fire vs no-fire)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
leg = ax.legend(frameon=False, markerscale=6, loc="upper right")
for t in leg.get_texts():
    t.set_color(INK2)
ax.grid(color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIGS / "fig2_spatial_distribution.png", bbox_inches="tight")
plt.close(fig)
print("fig2 done")

# ---------------------------------------------------------------- fig 3: variable distributions on last day
dist_vars = [
    ("t2m_c", "Max air temperature (°C)"),
    ("rh_pct", "Min relative humidity (%)"),
    ("wind_speed", "Max wind speed (m/s)"),
    ("tp_mm", "Total precipitation (mm)"),
    ("ndvi", "NDVI"),
    ("lai", "Leaf area index"),
    ("lst_day_c", "Day land-surface temp (°C)"),
    ("smi", "Soil moisture index"),
]
fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
for ax, (var, title) in zip(axes.ravel(), dist_vars):
    p = pos_last[var].dropna()
    n = neg_last[var].dropna()
    lo = min(p.quantile(0.01), n.quantile(0.01))
    hi = max(p.quantile(0.99), n.quantile(0.99))
    bins = np.linspace(lo, hi, 45)
    ax.hist(n, bins=bins, density=True, color=NOFIRE, alpha=0.55, label="No fire")
    ax.hist(p, bins=bins, density=True, color=FIRE, alpha=0.55, label="Fire")
    ax.set_title(title, fontsize=10.5)
    ax.set_yticks([])
    style_ax(ax)
    ax.grid(visible=False)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2,
           bbox_to_anchor=(0.99, 1.02))
fig.suptitle("Conditions on the day before the target day — fire vs no-fire",
             fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGS / "fig3_variable_distributions.png", bbox_inches="tight")
plt.close(fig)
print("fig3 done")

# ---------------------------------------------------------------- fig 4: 30-day lead-up trajectories
traj_vars = [
    ("t2m_c", "Max air temperature (°C)"),
    ("rh_pct", "Min relative humidity (%)"),
    ("smi", "Soil moisture index"),
    ("tp_mm", "Total precipitation (mm)"),
]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
days = np.arange(-30, 0)  # day -30 .. day -1 relative to target day
for ax, (var, title) in zip(axes, traj_vars):
    for df, color, lbl in ((neg, NOFIRE, "No fire"), (pos, FIRE, "Fire")):
        g = df.groupby("time_idx")[var]
        mean, sd = g.mean(), g.std()
        ax.plot(days, mean.values, color=color, linewidth=2, label=lbl)
        ax.fill_between(days, (mean - 0.3 * sd).values, (mean + 0.3 * sd).values,
                        color=color, alpha=0.15, linewidths=0)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("Days before target day")
    style_ax(ax)
axes[0].legend(frameon=False, loc="lower left", fontsize=9)
fig.suptitle("The 30-day lead-up to a fire day — mean trajectory (±0.3 SD band)",
             fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGS / "fig4_leadup_trajectories.png", bbox_inches="tight")
plt.close(fig)
print("fig4 done")

# ---------------------------------------------------------------- fig 5: correlation heatmap (diverging)
corr_vars = ["t2m", "d2m", "rh", "tp", "wind_speed", "ssrd", "sp",
             "lst_day", "lst_night", "ndvi", "lai", "smi",
             "dem", "slope", "population", "roads_distance"]
snap = pd.concat([pos_last, neg_last], ignore_index=True)
corr = snap[corr_vars].corr()

fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_vars)))
ax.set_yticks(range(len(corr_vars)))
ax.set_xticklabels(corr_vars, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr_vars, fontsize=9)
ax.set_title("Variable correlations (last observed day, all samples)")
ax.grid(visible=False)
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        v = corr.values[i, j]
        if abs(v) >= 0.5 and i != j:
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.75 else INK)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.outline.set_visible(False)
fig.tight_layout()
fig.savefig(FIGS / "fig5_correlation_heatmap.png", bbox_inches="tight")
plt.close(fig)
print("fig5 done")

# ---------------------------------------------------------------- fig 6: burned area sizes (positives)
fig, ax = plt.subplots(figsize=(8, 4))
ba = pos_last["burned_area_has"].clip(lower=1)
bins = np.logspace(np.log10(ba.min()), np.log10(ba.quantile(0.999)), 40)
ax.hist(ba, bins=bins, color=FIRE)
ax.set_xscale("log")
ax.set_xlabel("Final burned area (hectares, log scale)")
ax.set_ylabel("Fires")
ax.set_title("How large do the sampled fires get?")
med = ba.median()
ax.axvline(med, color=INK2, linewidth=1.5, linestyle="--")
ax.text(med * 1.15, ax.get_ylim()[1] * 0.9, f"median {med:,.0f} ha", color=INK2, fontsize=10)
style_ax(ax)
fig.tight_layout()
fig.savefig(FIGS / "fig6_burned_area_sizes.png", bbox_inches="tight")
plt.close(fig)
print("fig6 done")

# ---------------------------------------------------------------- headline stats for the report
def fmt_stats():
    s = {
        "n_pos": int(n_pos), "n_neg": int(n_neg),
        "years": [int(pos_last['time'].dt.year.min()), int(pos_last['time'].dt.year.max())],
        "pct_jul_aug": float((pos_last['time'].dt.month.isin([7, 8])).mean() * 100),
        "t2m_fire": float(pos_last['t2m_c'].mean()), "t2m_nofire": float(neg_last['t2m_c'].mean()),
        "rh_fire": float(pos_last['rh_pct'].mean()), "rh_nofire": float(neg_last['rh_pct'].mean()),
        "smi_fire": float(pos_last['smi'].mean()), "smi_nofire": float(neg_last['smi'].mean()),
        "ndvi_fire": float(pos_last['ndvi'].mean()), "ndvi_nofire": float(neg_last['ndvi'].mean()),
        "median_ba": float(pos_last['burned_area_has'].median()),
        "max_ba": float(pos_last['burned_area_has'].max()),
        "corr_lst_t2m": float(corr.loc['lst_day', 't2m']),
        "corr_ndvi_lai": float(corr.loc['ndvi', 'lai']),
        "corr_rh_d2m": float(corr.loc['rh', 'd2m']),
    }
    return s

stats = fmt_stats()
with open(FIGS / "eda_stats.json", "w") as f:
    json.dump(stats, f, indent=1)
print(json.dumps(stats, indent=1))
print("ALL DONE")
