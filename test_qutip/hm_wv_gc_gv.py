import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------
# Load results
# -----------------------------------------------------------
df = pd.read_csv("scan_results_wv_gc_gv.csv")

# Unique vibrational frequencies (sorted)
wv_values = sorted(df["wv"].unique())

# Set up subplots
ncols = len(wv_values)
fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4), sharey=True)

for i, wv_sel in enumerate(wv_values):
    ax = axes[i] if ncols > 1 else axes

    df_slice = df[np.isclose(df["wv"], wv_sel, atol=1e-6)]

    # Pivot to get a 2D grid of Fidelity vs gc, gv
    pivot = df_slice.pivot(index="gv", columns="gc", values="Fidelity")
    pivot = pivot.sort_index().sort_index(axis=1)

    im = ax.imshow(pivot.values,
                   extent=[pivot.columns.min(), pivot.columns.max(),
                           pivot.index.min(), pivot.index.max()],
                   origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=1)

    ax.set_title(f"wv = {wv_sel:.3f}")
    ax.set_xlabel("gc")
    if i == 0:
        ax.set_ylabel("gv")

fig.suptitle("Fidelity vs (gc, gv) at different vibrational frequencies", fontsize=14)
fig.colorbar(im, ax=axes.ravel().tolist(), label="Fidelity")

plt.tight_layout()
plt.show()

