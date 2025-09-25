import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv("scan_results.csv")

# Choose which gc value to show across panels
gc_sel = 0.01  # pick one of the values in your sweep

# Unique gv values (sorted)
gv_values = sorted(df["gv"].unique())

# Make subplots
ncols = len(gv_values)
fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4), sharey=True)

for i, gv_sel in enumerate(gv_values):
    ax = axes[i] if ncols > 1 else axes

    df_slice = df[(np.isclose(df["gc"], gc_sel, atol=1e-6)) &
                  (np.isclose(df["gv"], gv_sel, atol=1e-6))]

    # Pivot to make a T1 x T2 grid
    pivot = df_slice.pivot(index="T1", columns="T2", values="Fidelity")
    pivot = pivot.sort_index().sort_index(axis=1)

    im = ax.imshow(pivot.values,
                   extent=[pivot.columns.min(), pivot.columns.max(),
                           pivot.index.min(), pivot.index.max()],
                   origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=1)

    ax.set_title(f"gv = {gv_sel:.3f}")
    ax.set_xlabel("T2")
    if i == 0:
        ax.set_ylabel("T1")

fig.suptitle(f"Fidelity heatmaps at gc = {gc_sel:.3f}", fontsize=14)
fig.colorbar(im, ax=axes.ravel().tolist(), label="Fidelity")

plt.tight_layout()
plt.show()

