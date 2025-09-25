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

Nf = 2   # bosonic cutoff
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

# frequencies
omega_q = LiH_params["w_q1"]
omega_c = LiH_params["w_cav"]
omega_v = LiH_params["w_vib1"]

# bare Hamiltonians
H_q1 = tensor(-omega_q/2 * sz, Iq, Iv, Iv, Ic)
H_q2 = tensor(Iq, -omega_q/2 * sz, Iv, Iv, Ic)
H_v1 = tensor(Iq, Iq, omega_v * nc, Iv, Ic)
H_v2 = tensor(Iq, Iq, Iv, omega_v * nc, Ic)
H_cav = tensor(Iq, Iq, Iv, Iv, omega_c * nc)

H_bare_total = H_q1 + H_q2 + H_v1 + H_v2 + H_cav

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
gc_values = np.linspace(0.001, 0.1, 11)   # qubit–cavity coupling
gv_values = np.linspace(1e-4, 0.005, 11)   # qubit–vibration coupling

T_scale = np.array([0.8, 0.9, 1.0, 1.1, 1.2])     # gate time qubit 1


# time discretization
tlist = np.linspace(0, 1400, 5000)

# -----------------------------------------------------------
# Run sweep and save results
# -----------------------------------------------------------
with open("scan_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["gc", "gv", "T1", "T2", "Fidelity", "Concurrence"])

    for gc in gc_values:
        for gv in gv_values:
            for T1s in T_scale:
                for T2s in T_scale:

                    # interaction terms
                    H_q1_cav = gc * (tensor(sp, Iq, Iv, Iv, am) + tensor(sm, Iq, Iv, Iv, ap))
                    H_q2_cav = gc * (tensor(Iq, sp, Iv, Iv, am) + tensor(Iq, sm, Iv, Iv, ap))

                    H_q1_vib1 = gv * tensor(sp*sm, Iq, (am+ap), Iv, Ic)
                    H_q2_vib2 = gv * tensor(Iq, sp*sm, Iv, (am+ap), Ic)

                    H_bare_vc = H_bare_total + H_q1_vib1 + H_q2_vib2

                    # time windows (global variables used inside w1_t, w2_t)
                    global T0_1, T0_2, T_gate_1, T_gate_2
                    T0_1 = 20
                    T_gate_1 = T1s * (1*pi)/(4 * np.abs(gc))
                    T0_2 = T0_1 + T_gate_1
                    T_gate_2 = T2s * (2 *pi)/(4 * np.abs(gc))

                    # time-dependent Hamiltonian
                    H_t = [[H_bare_vc, wc_t],
                           [H_q1_cav, w1_t],
                           [H_q2_cav, w2_t]]

                    res = mesolve(H_t, psi0, tlist, [], e_ops=[])
                    rho_final = res.states[-1]

                    rho_qubits = ptrace(rho_final, [0,1])
                    fide = fidelity(rho_qubits, rho_qubits_ideal)
                    conc = concurrence(rho_qubits)

                    writer.writerow([gc, gv, T1s, T2s, fide, conc])
                    print(f"gc={gc:.3f}, gv={gv:.3f}, T1s={T1s:.1f}, T2s={T2s:.1f}, F={fide:.6f}, C={conc:.6f}")

