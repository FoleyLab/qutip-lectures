# Finite Dicke model: pilot code to study photon occupation and many-body gap scaling
# ω_c = ω_z = 1 a.u., λ = λ_c = 0.5 * sqrt(ω_c * ω_z) = 0.5
#
# H = ω_c a†a + ω_z S_z + (2λ/√N)(a + a†) S_x    (maximal-spin sector j = N/2)
# Critical coupling under this convention: λ_c = 0.5 * sqrt(ω_c * ω_z)
#
# The script:
#  - Builds H in a photon ⊗ spin-(j=N/2) basis
#  - Auto-converges the photon cutoff nmax
#  - Computes E0, E1, Δ, and ⟨n⟩ for N=5..10 at λ=λc
#  - Fits power laws Δ ~ N^{-α}, ⟨n⟩ ~ N^{β} (log–log)
#  - Saves a CSV and two plots

import numpy as np
import pandas as pd
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import os

def spin_operators_j(j: float):
    d = int(2 * j + 1)
    m_vals = np.linspace(-j, j, d)
    Sp = np.zeros((d, d), dtype=np.float64)
    for i, m in enumerate(m_vals[:-1]):
        Sp[i + 1, i] = np.sqrt(j * (j + 1) - m * (m + 1))
    Sm = Sp.T
    Sx = 0.5 * (Sp + Sm)
    Sz = np.diag(m_vals)
    return Sx, Sz

def boson_operators(nmax: int):
    dim = nmax + 1
    a = np.zeros((dim, dim), dtype=np.float64)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(n)
    adag = a.T
    n_op = adag @ a
    return a, adag, n_op

def dicke_hamiltonian(N: int, lamb: float, w_c: float = 1.0, w_z: float = 1.0, nmax: int = 10):
    j = N / 2.0
    Sx, Sz = spin_operators_j(j)
    a, adag, n_op = boson_operators(nmax)
    I_b = np.eye(nmax + 1)
    I_s = np.eye(Sx.shape[0])
    H = (np.kron(w_c * (adag @ a), I_s)
         + np.kron(I_b, w_z * Sz)
         + (2.0 * lamb / np.sqrt(N)) * np.kron((a + adag), Sx))
    Nph = np.kron(n_op, I_s)
    return H, Nph

def ground_state_observables(H, Nph):
    evals, evecs = eigh(H)
    E0, E1 = evals[0], evals[1]
    psi0 = evecs[:, 0]
    gap = E1 - E0
    n_expect = float(np.real(psi0.conj().T @ (Nph @ psi0)))
    return E0, E1, gap, n_expect

def converged_observables(N, lamb, w_c=1.0, w_z=1.0,
                          nmax_start=8, nmax_step=4, nmax_cap=40, tol=1e-6):
    last_gap, last_n = None, None
    nmax = nmax_start
    while True:
        H, Nph = dicke_hamiltonian(N, lamb, w_c=w_c, w_z=w_z, nmax=nmax)
        E0, E1, gap, n_expect = ground_state_observables(H, Nph)
        if last_gap is not None:
            if abs(gap - last_gap) < tol and abs(n_expect - last_n) < tol:
                return E0, E1, gap, n_expect, nmax
        if nmax >= nmax_cap:
            return E0, E1, gap, n_expect, nmax
        last_gap, last_n = gap, n_expect
        nmax += nmax_step

# Parameters
w_c = 1.0; w_z = 1.0
lambda_c = 0.5 * np.sqrt(w_c * w_z)  # = 0.5 here
Ns = list(range(5, 15))              # 5..10

# Compute and collect
rows = []
for N in Ns:
    E0, E1, gap, n_ph, nmax_used = converged_observables(N, lambda_c, w_c=w_c, w_z=w_z,
                                                         nmax_start=8, nmax_step=4, nmax_cap=48, tol=1e-7)
    rows.append(dict(N=N, E0=E0, E1=E1, gap=gap, n_ph=n_ph, nmax_used=nmax_used))
df = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)

# Fits (log–log)
logN = np.log(df["N"].values.astype(float))
b_gap, a_gap = np.polyfit(logN, np.log(df["gap"].values.astype(float)), 1)  # slope, intercept
b_nph, a_nph = np.polyfit(logN, np.log(df["n_ph"].values.astype(float)), 1)
alpha_est = -b_gap
beta_est  =  b_nph

# Save CSV
os.makedirs("data/", exist_ok=True)
df.to_csv("data/dicke_scaling_results.csv", index=False)

# Plots
plt.figure(figsize=(5.0, 4.0))
plt.loglog(df["N"], df["gap"], marker="o")
plt.xlabel("N (atoms)")
plt.ylabel("Many-body gap Δ (E1 - E0)")
plt.title(f"Gap scaling at λ = λc = {lambda_c:.3f}\nFit exponent α ≈ {alpha_est:.3f} (Δ ~ N^(-α))")
plt.tight_layout(); plt.savefig("data/dicke_gap_scaling.png", dpi=150); plt.show()

plt.figure(figsize=(5.0, 4.0))
plt.loglog(df["N"], df["n_ph"], marker="s")
plt.xlabel("N (atoms)")
plt.ylabel("Ground-state photon number ⟨n⟩")
plt.title(f"Photon scaling at λ = λc = {lambda_c:.3f}\nFit exponent β ≈ {beta_est:.3f} (⟨n⟩ ~ N^(β))")
plt.tight_layout(); plt.savefig("data/dicke_photon_scaling.png", dpi=150); plt.show()

print(df)
print(f"Fits: Δ ~ N^(-α) with α ≈ {alpha_est:.4f}; ⟨n⟩ ~ N^(β) with β ≈ {beta_est:.4f}")

