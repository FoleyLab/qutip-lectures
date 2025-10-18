# optimize_gates.py
import json
import argparse
import numpy as np
from numpy import array
from qutip import *
from qutip.core.gates import *
from scipy.optimize import differential_evolution

pi = np.pi
au_to_inv_cm = 219474.63  # conversion

# ---------------------------
# Utilities & small helpers
# ---------------------------
def read_json_to_dict(filename: str) -> dict:
    with open(filename, "r") as f:
        return json.load(f)

def ladder_operators(dim):
    a = destroy(dim)
    adag = a.dag()
    return a.full(), adag.full()

def _tensor(*ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def build_projector(vectors, i, j):
    """Given matrix 'vectors' whose columns are basis/eigenvectors, build |i><j|."""
    ket = vectors[:, i][:, np.newaxis]
    bra = vectors[:, j][:, np.newaxis].conj().T
    return ket @ bra

# ---------------------------
# Vibronic-coupling builders
# ---------------------------
def build_qubit_vibronic_coupling(omega_list, gv, boson_dim, qubit_number=1):
    """
    Builds H_vib = sum_i omega_i |i><i| \otimes n_v   +   gv (|0><1|+|1><0|)\otimes(a+a^\dagger)
    on the full space: q1 ⊗ cav ⊗ q2 ⊗ v1 ⊗ v2  (each vib space is size boson_dim)
    """
    # local identities (as numpy arrays)
    Iq_np = np.eye(2, dtype=complex)
    Ic_np = np.eye(boson_dim, dtype=complex)
    Iv_np = np.eye(boson_dim, dtype=complex)

    # vib ladders
    a_np, adag_np = ladder_operators(boson_dim)
    n_np = adag_np @ a_np

    total_dim = 2 * boson_dim * 2 * boson_dim * boson_dim
    H_vib = np.zeros((total_dim, total_dim), dtype=complex)

    for i in range(2):
        Proj_ii = build_projector(Iq_np, i, i)  # columns of identity = comp. basis
        if qubit_number == 1:
            op_n = _tensor(Proj_ii, Ic_np, Iq_np, n_np, Iv_np)
        else:
            op_n = _tensor(Iq_np, Ic_np, Proj_ii, Iv_np, n_np)
        H_vib += omega_list[i] * op_n

    # vibronic coupling: |1><1| x (a^+ + a)
    Proj_11 = build_projector(Iq_np, 1, 1)
    if qubit_number == 1:
        op_m = _tensor(Proj_11, Ic_np, Iq_np, (a_np + adag_np), Iv_np)
    else:
        op_m = _tensor(Iq_np, Ic_np, Proj_11, Iv_np, (a_np + adag_np))

    H_vib_coup = gv * op_m
    qubit_dims = [2, boson_dim, 2, boson_dim, boson_dim]
    return Qobj(H_vib + H_vib_coup, dims=[qubit_dims, qubit_dims])

def build_qubit_cavity_vibronic_coupling(H_qubit_cavity_coupling, omega_list, gv_list, boson_dim, qubit_number=1):
    """
    Builds vibronic coupling in the *polariton* basis obtained from H_qubit_cavity_coupling.
    """
    # Trace to (qubit,cavity) subspace for the chosen qubit
    if qubit_number == 1:
        H_qc_sub = H_qubit_cavity_coupling.ptrace([0, 1])
    else:
        H_qc_sub = H_qubit_cavity_coupling.ptrace([1, 2])

    H_coup_np = H_qc_sub.full()
    vals, vecs = np.linalg.eigh(H_coup_np)  # columns = eigenvectors (polaritons)
    polariton_dim = H_coup_np.shape[0]

    # operators on vib space
    a_np, adag_np = ladder_operators(boson_dim)
    n_np = adag_np @ a_np

    Iq_np = np.eye(2, dtype=complex)
    Iv_np = np.eye(boson_dim, dtype=complex)

    total_dim = polariton_dim * 2 * boson_dim * boson_dim
    H_vib = np.zeros((total_dim, total_dim), dtype=complex)
    H_vib_coup = np.zeros_like(H_vib)

    # H_vib: diagonal per polariton; H_vib_coup: |pol1><pol2|, |pol1><pol3|
    for i in range(3):  # pol1 (ground-like), pol2 (LP), pol3 (UP)
        Proj_ii = build_projector(vecs, i, i)
        if qubit_number == 1:
            op_n = _tensor(Proj_ii, Iq_np, n_np, Iv_np)
        else:
            op_n = _tensor(Iq_np, Proj_ii, Iv_np, n_np)
        H_vib += omega_list[i] * op_n

        if i > 0:
            #Proj_0i = build_projector(vecs, 0, i) + build_projector(vecs, i, 0)
            if qubit_number == 1:
                op_c = _tensor(Proj_ii, Iq_np, (a_np + adag_np), Iv_np)
            else:
                op_c = _tensor(Iq_np, Proj_ii, Iv_np, (a_np + adag_np))
            H_vib_coup += gv_list[i - 1] * op_c  

    qubit_dims = [2, boson_dim, 2, boson_dim, boson_dim]
    return Qobj(H_vib + H_vib_coup, dims=[qubit_dims, qubit_dims])

# ---------------------------
# Core simulation ("runner")
# ---------------------------
def simulate(LiH_params, psi0, rho_qubit_ideal, tlist, T_gate_1=None, T_gate_2=None,
             return_state=False):
    """
    One full forward simulation with specified T_gate_1,T_gate_2 (if None, use JC heuristics).
    Returns (rho_qubits, fidelity, concurrence).
    """
    # constants (atomic units)
    omega_q1 = LiH_params["w_q1"]
    omega_q2 = LiH_params["w_q2"]
    omega_c  = LiH_params["w_cav"]

    mu_eg = LiH_params["qubit_1_dipole_moments"]["mu_eg"]
    mu_ee = LiH_params["qubit_1_dipole_moments"]["mu_e"]
    mu_gg = LiH_params["qubit_1_dipole_moments"]["mu_g"]

    Nf = LiH_params.get("N_boson", 2)
    lz = LiH_params.get("lambda_z", 0.01)
    model = LiH_params.get("Model", "Pauli-Fierz")

    # cavity coupling strength (Pauli-Fierz/Rabi derived)
    g_cav = np.sqrt(omega_c / 2) * lz * abs(mu_eg)

    # vibrational constants (cm^-1 → a.u.)
    omega_v_S0   = LiH_params["w_vibS0_cm-1"] / au_to_inv_cm
    omega_v_S1   = LiH_params["w_vibS1_cm-1"] / au_to_inv_cm
    omega_v_pol1 = LiH_params["w_vib1_cm-1"]  / au_to_inv_cm
    omega_v_pol2 = LiH_params["w_vib2_cm-1"]  / au_to_inv_cm
    omega_v_pol3 = LiH_params["w_vib3_cm-1"]  / au_to_inv_cm

    qubit_omega_list     = np.array([omega_v_S0, omega_v_S1])
    polariton_omega_list = np.array([omega_v_pol1, omega_v_pol2, omega_v_pol3])

    # Huang-Rhys → vibronic couplings
    S_S0S1 = LiH_params["S_S0S1"]
    S_GLP  = LiH_params["S_GLP"]
    S_GUP  = LiH_params["S_GUP"]

    qubit_gv = np.sqrt(S_S0S1) * omega_v_S0
    polariton_gv = np.array([np.sqrt(S_GLP) * omega_v_pol1,
                             np.sqrt(S_GUP) * omega_v_pol1])

    # Default (heuristic) JC-like gate durations if not provided
    if T_gate_1 is None or T_gate_2 is None:
        T_gate_1 = pi / (4 * g_cav)
        T_gate_2 = pi / (2 * g_cav)

    # timing windows
    T0_1 = 20.0
    T0_2 = T0_1 + T_gate_1

    # dipole operator
    d_matrix_fixed = lz * np.array([[mu_gg, mu_eg], [mu_eg, mu_ee]], dtype=complex)
    d_qobj = Qobj(d_matrix_fixed)

    # local ops
    sz, sm, sp = sigmaz(), destroy(2), destroy(2).dag()
    nq = sp * sm
    am, ap = destroy(Nf), destroy(Nf).dag()
    nc = ap * am
    Iq, Ic, Iv = qeye(2), qeye(Nf), qeye(Nf)

    # expectation operators
    ncav = tensor(Iq, nc, Iq, Iv, Iv)
    nq1  = tensor(nq, Ic, Iq, Iv, Iv)
    nq2  = tensor(Iq, Ic, nq, Iv, Iv)

    # bare H
    H_q1  = tensor(-omega_q1 / 2 * sz, Ic, Iq, Iv, Iv)
    H_cav = tensor(Iq, omega_c * nc, Iq, Iv, Iv)
    H_q2  = tensor(Iq, Ic, -omega_q2 / 2 * sz, Iv, Iv)

    # light-matter coupling model
    if model == "Pauli-Fierz":
        H_q1_cav = -np.sqrt(omega_c / 2) * tensor(d_qobj, (am + ap), Iq, Iv, Iv) + 0.5 * tensor(d_qobj * d_qobj, Ic, Iq, Iv, Iv)
        H_q2_cav = -np.sqrt(omega_c / 2) * tensor(Iq, (am + ap), d_qobj, Iv, Iv) + 0.5 * tensor(Iq, Ic, d_qobj * d_qobj, Iv, Iv)
    elif model == "Rabi":
        H_q1_cav = -np.sqrt(omega_c / 2) * tensor(d_qobj, (am + ap), Iq, Iv, Iv)
        H_q2_cav = -np.sqrt(omega_c / 2) * tensor(Iq, (am + ap), d_qobj, Iv, Iv)
    elif model == "Jaynes-Cummings":
        g = g_cav
        H_q1_cav = g * tensor(sp, am, Iq, Iv, Iv) + g * tensor(sm, ap, Iq, Iv, Iv)
        H_q2_cav = g * tensor(Iq, am, sp, Iv, Iv) + g * tensor(Iq, ap, sm, Iv, Iv)
    else:
        # default to PF
        H_q1_cav = -np.sqrt(omega_c / 2) * tensor(d_qobj, (am + ap), Iq, Iv, Iv) + 0.5 * tensor(d_qobj * d_qobj, Ic, Iq, Iv, Iv)
        H_q2_cav = -np.sqrt(omega_c / 2) * tensor(Iq, (am + ap), d_qobj, Iv, Iv) + 0.5 * tensor(Iq, Ic, d_qobj * d_qobj, Iv, Iv)

    # vibronic terms
    H_q1_vib  = build_qubit_vibronic_coupling(qubit_omega_list, qubit_gv, Nf, qubit_number=1)
    H_q2_vib  = build_qubit_vibronic_coupling(qubit_omega_list, qubit_gv, Nf, qubit_number=2)
    H_pol1_vib = build_qubit_cavity_vibronic_coupling((H_q1_cav + H_q1 + H_cav), polariton_omega_list, polariton_gv, Nf, qubit_number=1)
    H_pol2_vib = build_qubit_cavity_vibronic_coupling((H_q2_cav + H_q2 + H_cav), polariton_omega_list, polariton_gv, Nf, qubit_number=2)

    # time windows
    def wc_t(t, args=None): return 1.0 if t <= T0_1 else 0.0
    def w1_t(t, args=None): return 1.0 if (T0_1 < t <= T0_1 + T_gate_1) else 0.0
    def w2_t(t, args=None): return 1.0 if (T0_2 < t <= T0_2 + T_gate_2) else 0.0

    H_uncoupled       = H_q1 + H_cav + H_q2 + H_q1_vib + H_q2_vib
    H_q1_cav_coupled  = H_q1 + H_cav + H_q1_cav + H_pol1_vib + H_q2 + H_q2_vib
    H_q2_cav_coupled  = H_q1 + H_cav + H_q1_vib + H_q2 + H_q2_cav + H_pol2_vib

    H_t = [[H_uncoupled, wc_t], [H_q1_cav_coupled, w1_t], [H_q2_cav_coupled, w2_t]]

    res = mesolve(H_t, psi0, tlist, [], e_ops=[])
    rho_final = res.states[-1]
    rho_qubits = ptrace(rho_final, [0, 2])

    F = fidelity(rho_qubits, rho_qubit_ideal)
    C = concurrence(rho_qubits)

    return (rho_qubits if return_state else None, float(F), float(C))

# ---------------------------
# Optimization wrapper
# ---------------------------
def optimize_gate_times(LiH_params, psi_init, rho_qubits_ideal, tlist,
                        objective="fidelity",
                        popsize=15, maxiter=40, seed=1234, workers=1):
    """
    Optimize (T_gate_1, T_gate_2) to maximize chosen objective.
    Bounds are set relative to the JC heuristic times derived from g_cav.
    """
    omega_c = LiH_params["w_cav"]
    mu_eg = LiH_params["qubit_1_dipole_moments"]["mu_eg"]
    lz = LiH_params.get("lambda_z", 0.01)
    g_cav = np.sqrt(omega_c / 2) * lz * abs(mu_eg)

    # JC heuristics
    T1_star = pi / (4 * g_cav)
    T2_star = pi / (2 * g_cav)

    # Reasonable search bounds (broad but finite)
    # You can tighten these if you already know your regime.
    bounds = [(0.2 * T1_star, 1.5 * T1_star),
              (0.2 * T2_star, 1.5 * T2_star)]

    # Make sure tlist extends beyond the latest possible end time
    latest_end = 20.0 + bounds[0][1] + bounds[1][1]
    if tlist[-1] < latest_end:
        raise ValueError(f"tlist[-1]={tlist[-1]} is too short for the search window (needs ≥ {latest_end:.3f}).")

    objective = objective.lower().strip()
    if objective not in ("fidelity", "concurrence"):
        raise ValueError("objective must be 'fidelity' or 'concurrence'")

    # define cost (minimize negative of target)
    def cost(x):
        T1, T2 = float(x[0]), float(x[1])
        _, F, C = simulate(LiH_params, psi_init, rho_qubits_ideal, tlist, T1, T2, return_state=False)
        return -F if objective == "fidelity" else -C

    result = differential_evolution(cost, bounds=bounds, strategy="best1bin",
                                    maxiter=maxiter, popsize=popsize,
                                    tol=1e-6, mutation=(0.5, 1.0), recombination=0.7,
                                    seed=seed, workers=workers, updating="deferred" if workers != 1 else "immediate")

    T1_opt, T2_opt = result.x
    _, F_opt, C_opt = simulate(LiH_params, psi_init, rho_qubits_ideal, tlist, T1_opt, T2_opt, return_state=False)

    return {
        "T_gate_1_opt": float(T1_opt),
        "T_gate_2_opt": float(T2_opt),
        "Max Time" : float(tlist[-1]),
        "fidelity_at_opt": float(F_opt),
        "concurrence_at_opt": float(C_opt),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message)
    }

# ---------------------------
# Script entry point
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Optimize T_gate_1 and T_gate_2 for fidelity or concurrence.")
    parser.add_argument("param_json", type=str, help="Path to parameter JSON (all other params read from here).")
    parser.add_argument("--objective", type=str, default="fidelity", choices=["fidelity", "concurrence"],
                        help="Optimization objective.")
    parser.add_argument("--n_boson_override", type=int, default=None, help="Optionally override N_boson from JSON.")
    parser.add_argument("--tmax", type=float, default=None, help="Override max simulation time (a.u.).")
    parser.add_argument("--nt", type=int, default=5000, help="Number of time steps in tlist.")
    parser.add_argument("--popsize", type=int, default=15, help="DE popsize.")
    parser.add_argument("--maxiter", type=int, default=40, help="DE max generations.")
    parser.add_argument("--seed", type=int, default=1234, help="DE seed.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for DE (1 = serial).")
    args = parser.parse_args()

    LiH_params = read_json_to_dict(args.param_json)

    # Optionally override vib Fock dim
    if args.n_boson_override is not None:
        LiH_params["N_boson"] = int(args.n_boson_override)

    # Prepare initial state, ideal target, and tlist
    Nf = LiH_params.get("N_boson", 3)

    psi_init = tensor(basis(2,1), basis(Nf,0),
                      basis(2,0), basis(Nf,0), basis(Nf,0))

    rho_qubits_ideal = ket2dm(
        tensor(phasegate(0), phasegate(pi/2)) *
        sqrtiswap() *
        tensor(basis(2,1), basis(2,0))
    )

    # Build a conservative tlist if not provided: ensure long enough for broad bounds
    omega_c  = LiH_params["w_cav"]
    mu_eg    = LiH_params["qubit_1_dipole_moments"]["mu_eg"]
    lz       = LiH_params.get("lambda_z", 0.01)
    g_cav    = np.sqrt(omega_c / 2) * lz * abs(mu_eg)

    T1_star = pi / (4 * g_cav)
    T2_star = pi / (2 * g_cav)
    # End time: start(20) + 5*T1* + 5*T2* plus margin
    tmax_default = 20.0 + 5.0 * T1_star + 5.0 * T2_star + 10.0
    tmax = args.tmax if args.tmax is not None else tmax_default

    tlist = np.linspace(0.0, float(tmax), int(args.nt))
    

    # Run the optimization
    result = optimize_gate_times(LiH_params, psi_init, rho_qubits_ideal, tlist,
                                 objective=args.objective, popsize=args.popsize,
                                 maxiter=args.maxiter, seed=args.seed, workers=args.workers)

    # Report
    print("\n=== Optimization Result ===")
    print(f"Objective:         {args.objective}")
    print(f"Success:           {result['success']}")
    print(f"Message:           {result['message']}")
    print(f"Iterations:        {result['nit']}")
    print(f"Func evals:        {result['nfev']}")
    print(f"T_gate_1_opt (au): {result['T_gate_1_opt']:.8f}")
    print(f"T_gate_2_opt (au): {result['T_gate_2_opt']:.8f}")
    print(f"Max Time (a.u.):   {result['Max Time']:.8f}")
    print(f"Fidelity @ opt:    {result['fidelity_at_opt']:.10f}")
    print(f"Concurrence @ opt: {result['concurrence_at_opt']:.10f}")

if __name__ == "__main__":
    main()

