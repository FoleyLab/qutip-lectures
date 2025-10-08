#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan over gv1, gv2 and optimize T1_gate, T2_gate for maximum fidelity or concurrence.
Combines vibronic runner() with global (differential evolution) + local (L-BFGS-B) optimization.
"""
import numpy as np
from qutip import qeye, destroy, tensor, Qobj, basis, sigmaz, mesolve, fidelity, concurrence
from qutip.core.gates import sqrtiswap, phasegate
from qutip.core.gates import *
from scipy.optimize import minimize, differential_evolution
import json, csv
import matplotlib.pyplot as plt

pi = np.pi

# =======================================================================
# ------------------------- Utility functions ---------------------------
# =======================================================================

def read_json_to_dict(filename: str) -> dict:
    with open(filename, "r") as f:
        return json.load(f)

def ladder_operators(dim):
    a = destroy(dim)
    adag = a.dag()
    return a.full(), adag.full()

def _tensor(*ops):
    """Kronecker product of multiple NumPy operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def build_projector(vectors, i, j):
    """|i><j| from eigenvectors."""
    ket = vectors[:, i][:, np.newaxis]
    bra = vectors[:, j][:, np.newaxis].conj().T
    return ket @ bra

def build_qubit_cavity_vibronic_coupling(H_qubit_cavity_coupling, omega_list, gv_list, boson_dim, qubit_number=1):
    """Construct vibronic coupling Hamiltonian projected onto polaritonic subspace."""
    if qubit_number == 1:
        H_qubit_cavity_coupling = H_qubit_cavity_coupling.ptrace([0,1])
    elif qubit_number == 2:
        H_qubit_cavity_coupling = H_qubit_cavity_coupling.ptrace([1,2])

    H_coup_np = H_qubit_cavity_coupling.full()
    vals, vecs = np.linalg.eigh(H_coup_np)
    polariton_dim = H_coup_np.shape[0]

    a_np, adag_np = ladder_operators(boson_dim)
    n_np = adag_np @ a_np
    Iq_np = np.eye(2, dtype=complex)
    Iv_np = np.eye(boson_dim, dtype=complex)

    total_dim = polariton_dim * 2 * boson_dim * boson_dim
    H_vib = np.zeros((total_dim, total_dim), dtype=complex)
    H_vib_coup = np.zeros_like(H_vib)

    for i in range(polariton_dim):
        Proj_ii = build_projector(vecs, i, i)
        if qubit_number == 1:
            op_n = _tensor(Proj_ii, Iq_np, n_np, Iv_np)
        else:
            op_n = _tensor(Iq_np, Proj_ii, Iv_np, n_np)
        H_vib += omega_list[i] * op_n

        if i > 0:
            Proj_0i = build_projector(vecs, 0, i) + build_projector(vecs, i, 0)
            if qubit_number == 1:
                op_c = _tensor(Proj_0i, Iq_np, (a_np + adag_np), Iv_np)
            else:
                op_c = _tensor(Iq_np, Proj_0i, Iv_np, (a_np + adag_np))
            H_vib_coup = gv_list[i] * op_c

    qubit_dims = [2, boson_dim, 2, boson_dim, boson_dim]
    return Qobj(H_vib + H_vib_coup, dims=[qubit_dims, qubit_dims])

# =======================================================================
# ------------------------- Runner function -----------------------------
# =======================================================================

def runner(params, LiH_params, psi0, rho_qubits_ideal, tlist, Nf, omega_vib_list):
    """Compute fidelity and concurrence for given gv1, gv2, T1_gate, T2_gate."""
    gv_list = np.zeros_like(omega_vib_list)
    gv_list[1] = params[0]  # gv1 (LP)
    gv_list[2] = params[1]  # gv2 (UP)
    gv_list[4] = params[0] * 0
    gv_list[5] = params[1] * 0
    T_gate_1, T_gate_2 = params[2], params[3]

    global T0_1, T0_2
    T0_1 = 20
    T0_2 = T0_1 + T_gate_1

    # --- physical constants ---
    omega_q = LiH_params["w_q1"]
    omega_c = omega_q
    omega_v = LiH_params["w_vib1"]
    mu_eg = LiH_params["qubit_1_dipole_moments"]["mu_eg"]
    mu_ee = LiH_params["qubit_1_dipole_moments"]["mu_e"]
    mu_gg = LiH_params["qubit_1_dipole_moments"]["mu_g"]
    lv = 0.01

    gc_fixed = np.sqrt(omega_c / 2) * lv * np.abs(mu_eg)
    d_matrix_fixed = lv * np.array([[mu_gg, mu_eg], [mu_eg, mu_ee]])
    d_qobj = Qobj(d_matrix_fixed)

    # --- local ops ---
    sz, sm, sp = sigmaz(), destroy(2), destroy(2).dag()
    nq = sp * sm
    am, ap = destroy(Nf), destroy(Nf).dag()
    nc = ap * am
    Iq, Ic, Iv = qeye(2), qeye(Nf), qeye(Nf)

    # --- bare H parts ---
    H_q1 = tensor(-omega_q / 2 * sz, Ic, Iq, Iv, Iv)
    H_cav = tensor(Iq, omega_c * nc, Iq, Iv, Iv)
    H_q2 = tensor(Iq, Ic, -omega_q / 2 * sz, Iv, Iv)
    H_v1 = tensor(Iq, Ic, Iq, omega_v * nc, Iv)
    H_v2 = tensor(Iq, Ic, Iq, Iv, omega_v * nc)
    H_q1_cav_JC = gc_fixed * (tensor(sp, am, Iq, Iv, Iv) + tensor(sm, ap, Iq, Iv, Iv))
    H_q2_cav_JC = gc_fixed * (tensor(Iq, am, sp, Iv, Iv) + tensor(Iq, ap, sm, Iv, Iv))
    H_q1_v1 = gv_list[4] * tensor(sp * sm, Ic, Iq, (am + ap), Iv)
    H_q2_v2 = gv_list[4] * tensor(Iq, Ic, sp * sm, Iv, (am + ap))

    # --- vibronic coupling ---
    H_q1_vib1_coupled = build_qubit_cavity_vibronic_coupling(H_q1 + H_cav + H_q1_cav_JC,
                                                             omega_vib_list, gv_list, Nf, qubit_number=1)
    H_q2_vib2_coupled = build_qubit_cavity_vibronic_coupling(H_q2 + H_cav + H_q2_cav_JC,
                                                             omega_vib_list, gv_list, Nf, qubit_number=2)

    # --- time-dependence ---
    def wc_t(t, args=None): return 1 if t <= T0_1 else 0
    def w1_t(t, args=None): return 1 if T0_1 < t <= T0_1 + T_gate_1 else 0
    def w2_t(t, args=None): return 1 if T0_2 < t <= T0_2 + T_gate_2 else 0

    H_uncoupled = H_q1 + H_cav + H_q2 + H_v1 + H_v2 + H_q1_v1 + H_q2_v2
    H_q1_cav_coupled = H_q1 + H_cav + H_q1_cav_JC + H_q2 + H_v1 + H_v2 + H_q1_vib1_coupled + H_q2_v2
    H_q2_cav_coupled = H_q1 + H_cav + H_q2_cav_JC + H_q2 + H_v1 + H_v2 + H_q1_v1 + H_q2_vib2_coupled

    H_t = [[H_uncoupled, wc_t], [H_q1_cav_coupled, w1_t], [H_q2_cav_coupled, w2_t]]

    res = mesolve(H_t, psi0, tlist, [], e_ops=[])
    rho_final = res.states[-1]
    rho_qubits = rho_final.ptrace([0, 2])
    fide = fidelity(rho_qubits, rho_qubits_ideal)
    conc = concurrence(rho_qubits)
    return rho_qubits, fide, conc

# =======================================================================
# ---------------------- Optimization and Scan --------------------------
# =======================================================================

def objective_T(params, gv1, gv2, LiH_params, psi0, rho_qubits_ideal, tlist, Nf, omega_vib_list, target="product"):
    gv_params = [gv1, gv2, params[0], params[1]]
    _, fid, conc = runner(gv_params, LiH_params, psi0, rho_qubits_ideal, tlist, Nf, omega_vib_list)
    if target == "fidelity":
        score = fid
    elif target == "concurrence":
        score = conc
    elif target == "product":
        score = fid * conc
    else:
        raise ValueError("target must be 'fidelity', 'concurrence', or 'product'")
    return -score

def scan_gv_grid(LiH_params, psi0, rho_qubits_ideal, tlist, Nf, omega_vib_list,
                 gv1_range=(0.6, 0.9), gv2_range=(0.5, 0.8), n_points=3, target="product"):
    gv1_vals = np.linspace(*gv1_range, n_points)
    gv2_vals = np.linspace(*gv2_range, n_points)
    fidelities = np.zeros((n_points, n_points))
    concurrences = np.zeros_like(fidelities)

    with open("optimization_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gv1", "gv2", "T1", "T2", "Fidelity", "Concurrence"])
        for i, gv1 in enumerate(gv1_vals):
            for j, gv2 in enumerate(gv2_vals):
                print(f"\n>>> Optimizing for gv1={gv1:.3f}, gv2={gv2:.3f}")
                bounds = [(100, 800), (100, 800)]
                result_de = differential_evolution(
                    objective_T, bounds,
                    args=(gv1, gv2, LiH_params, psi0, rho_qubits_ideal, tlist, Nf, omega_vib_list, target),
                    maxiter=8, popsize=5, polish=False
                )
                result_lbfgs = minimize(
                    objective_T, result_de.x,
                    args=(gv1, gv2, LiH_params, psi0, rho_qubits_ideal, tlist, Nf, omega_vib_list, target),
                    method="L-BFGS-B", bounds=bounds, jac=False
                )
                gv_params = [gv1, gv2, result_lbfgs.x[0], result_lbfgs.x[1]]
                _, fid, conc = runner(gv_params, LiH_params, psi0, rho_qubits_ideal, tlist, Nf, omega_vib_list)
                fidelities[i, j] = fid
                concurrences[i, j] = conc
                writer.writerow([gv1, gv2, result_lbfgs.x[0], result_lbfgs.x[1], fid, conc])
                print(f"→ T1={result_lbfgs.x[0]:.2f}, T2={result_lbfgs.x[1]:.2f}, F={fid:.4f}, C={conc:.4f}")
    return gv1_vals, gv2_vals, fidelities, concurrences

# =======================================================================
# ------------------------------ Main ----------------------------------
# =======================================================================

if __name__ == "__main__":

    LiH_params = read_json_to_dict("LiH_params.json")
    omega_vib_list = [0.006]*6
    Nf = 3
    psi_init = tensor(basis(2,1), basis(Nf,0), basis(2,0), basis(Nf,0), basis(Nf,0))

    psi_bell = np.sqrt(1/2) * tensor(basis(2,1), basis(2,0)) - np.sqrt(1/2) * tensor(basis(2,0), basis(2,1))

    rho_qubits_ideal = psi_bell * psi_bell.dag()  #ket2dm(tensor(phasegate(0), phasegate(pi/2)) * sqrtiswap() *
                                                  # tensor(basis(2,1), basis(2,0)))
    tlist = np.linspace(0, 1400, 5000)

    gv1_vals, gv2_vals, fidelities, concurrences = scan_gv_grid(
        LiH_params, psi_init, rho_qubits_ideal, tlist, Nf, omega_vib_list,
        gv1_range=(0.01, 2.0), gv2_range=(0.01, 2.0), n_points=5, target="fidelity"
    )

    # --- Plot results ---
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(fidelities, origin="lower",
               extent=[gv2_vals[0], gv2_vals[-1], gv1_vals[0], gv1_vals[-1]],
               aspect="auto", cmap="viridis")
    plt.colorbar(label="Fidelity")
    plt.xlabel("gv2"); plt.ylabel("gv1")

    plt.subplot(1,2,2)
    plt.imshow(concurrences, origin="lower",
               extent=[gv2_vals[0], gv2_vals[-1], gv1_vals[0], gv1_vals[-1]],
               aspect="auto", cmap="magma")
    plt.colorbar(label="Concurrence")
    plt.xlabel("gv2"); plt.ylabel("gv1")

    plt.tight_layout()
    plt.show()

