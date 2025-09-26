import numpy as np
from qutip import *
from qutip.core.gates import *
import matplotlib.pyplot as plt
import json
import csv

pi = np.pi

# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------

def wc_t(t, args=None):
    return 1

def w1_t(t, args=None):
    if t > T0_1 and t <= T0_1 + T_gate_1:
        return 1
    else:
        return 0

def w2_t(t, args=None):
    if t > T0_2 and t <= T0_2 + T_gate_2:
        return 1
    else:
        return 0

def read_json_to_dict(filename: str) -> dict:
    with open(filename, 'r') as f:
        return json.load(f)

# -----------------------------------------------------------
# Load system parameters (LiH example)
# -----------------------------------------------------------
LiH_params = read_json_to_dict("LiH_params.json")

Nf = 4   # bosonic cutoff (increase if needed)

# local operators
sz = sigmaz()
sm = destroy(2)
sp = sm.dag()
nq = sp * sm

am = destroy(Nf)
ap = am.dag()
nc = ap * am

# identities
Iq = qeye(2)
Ic = qeye(Nf)
Iv = qeye(Nf)

# initial state: |e,g,0,0,0>
psi0 = tensor(basis(2,1), basis(2,0), basis(Nf,0), basis(Nf,0), basis(Nf,0))

# ideal Bell target
rho_qubits_ideal = ket2dm(
    tensor(phasegate(0), phasegate(pi/2)) *
    sqrtiswap() *
    tensor(basis(2,1), basis(2,0))
)

# -----------------------------------------------------------
# Parameter grids
# -----------------------------------------------------------
wqc_values = np.linspace(0.05, 0.2, 6)     # qubit/cavity frequencies (in resonance)
wv_values = np.linspace(0.001, 0.009, 10)  # vibrational frequencies
gc_values = np.linspace(0.01, 0.5, 10)    # qubit–cavity coupling
gv_values = np.linspace(0.0005, 0.1, 10)   # qubit–vibration coupling

# time discretization
tlist = np.linspace(0, 1400, 5000)

# -----------------------------------------------------------
# Run sweep and save results
# -----------------------------------------------------------
with open("scan_results_wqc_wv_gc_gv.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["wqc", "wv", "gc", "gv", "Fidelity", "Concurrence"])

    for wqc in wqc_values:
        for wv in wv_values:
            # bare Hamiltonians (update with current wqc and wv)
            H_q1 = tensor(-wqc/2 * sz, Iq, Iv, Iv, Ic)
            H_q2 = tensor(Iq, -wqc/2 * sz, Iv, Iv, Ic)
            H_v1 = tensor(Iq, Iq, wv * nc, Iv, Ic)
            H_v2 = tensor(Iq, Iq, Iv, wv * nc, Ic)
            H_cav = tensor(Iq, Iq, Iv, Iv, wqc * nc)

            H_bare_total = H_q1 + H_q2 + H_v1 + H_v2 + H_cav

            for gc in gc_values:
                for gv in gv_values:

                    # interaction terms
                    H_q1_cav = gc * (tensor(sp, Iq, Iv, Iv, am) + tensor(sm, Iq, Iv, Iv, ap))
                    H_q2_cav = gc * (tensor(Iq, sp, Iv, Iv, am) + tensor(Iq, sm, Iv, Iv, ap))

                    H_q1_vib1 = gv * tensor(sp*sm, Iq, (am+ap), Iv, Ic)
                    H_q2_vib2 = gv * tensor(Iq, sp*sm, Iv, (am+ap), Ic)

                    H_bare_vc = H_bare_total + H_q1_vib1 + H_q2_vib2

                    # gate times fixed by gc
                    global T0_1, T0_2, T_gate_1, T_gate_2
                    T0_1 = 20
                    T_gate_1 = pi / (4 * np.abs(gc))
                    T0_2 = T0_1 + T_gate_1
                    T_gate_2 = pi / (2 * np.abs(gc))

                    # time-dependent Hamiltonian
                    H_t = [[H_bare_vc, wc_t],
                           [H_q1_cav, w1_t],
                           [H_q2_cav, w2_t]]

                    res = mesolve(H_t, psi0, tlist, [], e_ops=[])
                    rho_final = res.states[-1]

                    rho_qubits = ptrace(rho_final, [0,1])
                    fide = fidelity(rho_qubits, rho_qubits_ideal)
                    conc = concurrence(rho_qubits)

                    writer.writerow([wqc, wv, gc, gv, fide, conc])
                    print(f"wqc={wqc:.3f}, wv={wv:.3f}, gc={gc:.3f}, gv={gv:.4f}, F={fide:.6f}, C={conc:.6f}")

