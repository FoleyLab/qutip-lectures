import numpy as np
from qutip import *
from qutip.core.gates import *
from scipy.optimize import minimize, differential_evolution
import json
import numpy as np

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
mu_eg = LiH_params["qubit_1_dipole_moments"]["mu_eg"]

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

# fixed frequencies
omega_q = LiH_params["w_q1"]
omega_c = omega_q  # enforce resonance
omega_v = LiH_params["w_vib1"]

# initial state: |e,g,0,0,0>
psi0 = tensor(basis(2,1), basis(2,0), basis(Nf,0), basis(Nf,0), basis(Nf,0))

# ideal Bell target
rho_qubits_ideal = ket2dm(
    tensor(phasegate(0), phasegate(pi/2)) *
    sqrtiswap() *
    tensor(basis(2,1), basis(2,0))
)

# time discretization
tlist = np.linspace(0, 1400, 5000)

# bare Hamiltonian (independent of gc, gv)
H_q1 = tensor(-omega_q/2 * sz, Iq, Iv, Iv, Ic)
H_q2 = tensor(Iq, -omega_q/2 * sz, Iv, Iv, Ic)
H_v1 = tensor(Iq, Iq, omega_v * nc, Iv, Ic)
H_v2 = tensor(Iq, Iq, Iv, omega_v * nc, Ic)
H_cav = tensor(Iq, Iq, Iv, Iv, omega_v * nc)
H_bare_total = H_q1 + H_q2 + H_v1 + H_v2 + H_cav



from scipy.optimize import minimize, differential_evolution
import numpy as np
from qutip import *

# -----------------------------------------------------------
# Flexible objective configuration
# -----------------------------------------------------------
objective_mode = "weighted"   # options: "product" or "weighted"
alpha, beta = 0.5, 0.5       # only used if mode="weighted"

# -----------------------------------------------------------
# Objective function
# -----------------------------------------------------------
def objective(params):
    gc, gv = params

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

    # store for inspection
    objective.last_fid = fide
    objective.last_conc = conc

    # choose objective type
    if objective_mode == "product":
        score = fide * conc
    elif objective_mode == "weighted":
        score = alpha * fide + beta * conc
    else:
        raise ValueError("objective_mode must be 'product' or 'weighted'")

    return -score  # minimize negative to maximize

# -----------------------------------------------------------
# Optimization pipeline
# -----------------------------------------------------------
#gv_min = 1e-5
#gv_max = 100 * gv_min
#gc_min = 10 * gv_max
#gc_max = 500 * gv_max

gc_min = 0.100
gc_max = 0.200

# min gv to get gc/gv ~ 230
gv_min = gc_min / 250
gv_max = gc_max / 10
bounds = [(gc_min, gc_max), (gv_min, gv_max)]

# Step 1: Global search with Differential Evolution
de_result = differential_evolution(objective, bounds,init="random", maxiter=100, popsize=500, polish=False)

print("\nGlobal (Differential Evolution) result:")
print(f"gc ≈ {de_result.x[0]:.5f}, gv ≈ {de_result.x[1]:.5f}")
print(f"gc / gv ≈ {de_result.x[0]/de_result.x[1]:.4f}")
objective(de_result.x)
print(f"Fidelity ≈ {objective.last_fid:.6f}, Concurrence ≈ {objective.last_conc:.6f}")

# Step 2: Local refinement with L-BFGS-B
lbfgs_result = minimize(objective, de_result.x, method="L-BFGS-B", bounds=bounds, jac=False)

print("\nLocal (L-BFGS-B) refinement:")
print(f"gc* = {lbfgs_result.x[0]:.5f}, gv* = {lbfgs_result.x[1]:.5f}")
print(f"gc* / gv* = {lbfgs_result.x[0]/lbfgs_result.x[1]:.4f}")
objective(lbfgs_result.x)
print(f"Fidelity* = {objective.last_fid:.6f}, Concurrence* = {objective.last_conc:.6f}")

if objective_mode == "product":
    print(f"Objective (F*C)* = {objective.last_fid*objective.last_conc:.6f}")
elif objective_mode == "weighted":
    print(f"Objective (αF + βC)* = {alpha*objective.last_fid + beta*objective.last_conc:.6f}")

