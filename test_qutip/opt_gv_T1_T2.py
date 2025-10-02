import numpy as np
from qutip import *
from qutip import Qobj
from qutip.core.gates import *
from scipy.optimize import minimize, differential_evolution
import json

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
# Load system parameters
# -----------------------------------------------------------
LiH_params = read_json_to_dict("LiH_params.json")

Nf = 4   # bosonic cutoff
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

# frequencies and dipoles
omega_q = LiH_params["w_q1"]
omega_c = omega_q  # resonance
omega_v = LiH_params["w_vib1"]
mu_eg = LiH_params["qubit_1_dipole_moments"]["mu_eg"]
mu_ee = LiH_params["qubit_1_dipole_moments"]["mu_e"]
mu_gg = LiH_params["qubit_1_dipole_moments"]["mu_g"]

# initial state: |e,g,0,0,0>
psi0 = tensor(basis(2,1), basis(2,0),
              basis(Nf,0), basis(Nf,0), basis(Nf,0))

# ideal Bell target
rho_qubits_ideal = ket2dm(
    tensor(phasegate(0), phasegate(pi/2)) *
    sqrtiswap() *
    tensor(basis(2,1), basis(2,0))
)

# time discretization
tlist = np.linspace(0, 1400, 5000)

# bare Hamiltonian
H_q1 = tensor(-omega_q/2 * sz, Iq, Iv, Iv, Ic)
H_q2 = tensor(Iq, -omega_q/2 * sz, Iv, Iv, Ic)
H_v1 = tensor(Iq, Iq, omega_v * nc, Iv, Ic)
H_v2 = tensor(Iq, Iq, Iv, omega_v * nc, Ic)
H_cav = tensor(Iq, Iq, Iv, Iv, omega_c * nc)
H_bare_total = H_q1 + H_q2 + H_v1 + H_v2 + H_cav

# dipole operator (fixed by lv)
lv = 0.05
gc_fixed = np.sqrt(omega_c / 2) * lv * np.abs(mu_eg)
d_matrix_fixed = lv * np.array([[mu_gg, mu_eg], [mu_eg, mu_ee]])
d_qobj = Qobj(d_matrix_fixed)

# -----------------------------------------------------------
# Build Hamiltonian terms depending on model
# -----------------------------------------------------------
def build_cavity_terms(model, gv):
    """Return H_q1_cav, H_q2_cav for a given model and gv"""
    if model == "Jaynes-Cummings":
        H_q1_cav = gc_fixed * (tensor(sp, Iq, Iv, Iv, am) +
                               tensor(sm, Iq, Iv, Iv, ap))
        H_q2_cav = gc_fixed * (tensor(Iq, sp, Iv, Iv, am) +
                               tensor(Iq, sm, Iv, Iv, ap))
    elif model == "Rabi":
        H_q1_cav = np.sqrt(omega_c / 2) * tensor(d_qobj, Iq, Iv, Iv, (am + ap))
        H_q2_cav = np.sqrt(omega_c / 2) * tensor(Iq, d_qobj, Iv, Iv, (am + ap))
    elif model == "Pauli-Fierz":
        H_q1_cav = (np.sqrt(omega_c / 2) * tensor(d_qobj, Iq, Iv, Iv, (am + ap)) +
                    0.5 * tensor(d_qobj * d_qobj, Iq, Iv, Iv, Ic))
        H_q2_cav = (np.sqrt(omega_c / 2) * tensor(Iq, d_qobj, Iv, Iv, (am + ap)) +
                    0.5 * tensor(Iq, d_qobj * d_qobj, Iv, Iv, Ic))
    else:
        raise ValueError("model must be 'Jaynes-Cummings', 'Rabi', or 'Pauli-Fierz'")
    return H_q1_cav, H_q2_cav

# -----------------------------------------------------------
# Objective function for optimization
# -----------------------------------------------------------
objective_mode = "product"   # "product" or "weighted"
alpha, beta = 0.5, 0.5       # for weighted sum

def objective(params, model="Jaynes-Cummings"):
    gv, T1, T2 = params

    # interaction terms
    H_q1_cav, H_q2_cav = build_cavity_terms(model, gv)
    H_q1_vib1 = gv * tensor(sp*sm, Iq, (am+ap), Iv, Ic)
    H_q2_vib2 = gv * tensor(Iq, sp*sm, Iv, (am+ap), Ic)

    H_bare_vc = H_bare_total + H_q1_vib1 + H_q2_vib2

    # set global gate times
    global T0_1, T0_2, T_gate_1, T_gate_2
    T0_1 = 20
    T_gate_1 = T1
    T0_2 = T0_1 + T1
    T_gate_2 = T2

    H_t = [[H_bare_vc, wc_t], [H_q1_cav, w1_t], [H_q2_cav, w2_t]]

    res = mesolve(H_t, psi0, tlist, [], e_ops=[])
    rho_final = res.states[-1]
    rho_qubits = ptrace(rho_final, [0,1])

    fide = fidelity(rho_qubits, rho_qubits_ideal)
    conc = concurrence(rho_qubits)

    objective.last_fid = fide
    objective.last_conc = conc

    if objective_mode == "product":
        score = fide * conc
    elif objective_mode == "weighted":
        score = alpha * fide + beta * conc
    else:
        raise ValueError("objective_mode must be 'product' or 'weighted'")

    return -score

# -----------------------------------------------------------
# Optimization wrapper
# -----------------------------------------------------------
def optimize_model(model="Jaynes-Cummings"):
    bounds = [
        (gc_fixed/10000, gc_fixed/2),  # gv
        (pi / (8 * np.abs(gc_fixed)), pi / (2 * np.abs(gc_fixed))),  # T1 range
        (pi / (4 * np.abs(gc_fixed)), pi / (1 * np.abs(gc_fixed)))   # T2 range
    ]

    # Step 1: Global optimization
    de_result = differential_evolution(objective, bounds, args=(model,),
                                       maxiter=15, popsize=5, polish=False)

    print(f"\n[Global Search - {model}]")
    print(f"gv ≈ {de_result.x[0]:.5f}, T1 ≈ {de_result.x[1]:.2f}, T2 ≈ {de_result.x[2]:.2f}")
    objective(de_result.x, model)
    print(f"Fidelity ≈ {objective.last_fid:.6f}, Concurrence ≈ {objective.last_conc:.6f}")

    # Step 2: Local refinement
    lbfgs_result = minimize(objective, de_result.x, args=(model,),
                            method="L-BFGS-B", bounds=bounds, jac=False)

    print(f"\n[Local Refinement - {model}]")
    print(f"gv* = {lbfgs_result.x[0]:.5f}, T1* = {lbfgs_result.x[1]:.2f}, T2* = {lbfgs_result.x[2]:.2f}")
    objective(lbfgs_result.x, model)
    print(f"Fidelity* = {objective.last_fid:.6f}, Concurrence* = {objective.last_conc:.6f}")

    return lbfgs_result

# Example: run for Rabi model
result = optimize_model("Pauli-Fierz")

