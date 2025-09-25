import numpy as np
from qutip import *
from qutip.core.gates import *
from numpy import array
from numpy import real
import matplotlib.pyplot as plt
pi = np.pi
import json 

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
    """
    Reads a JSON file and returns its contents as a dictionary.
    
    Parameters:
    - filename (str): Path to the input JSON file.
    
    Returns:
    - dict: Contents of the JSON file.
    """
    with open(filename, 'r') as f:
        return json.load(f)

LiH_params = read_json_to_dict("LiH_params.json")

# Number of bosonic Fock states
Nf = 5
# local qubit operators
sz = sigmaz()
sm = destroy(2)
sp = sm.dag()
# qubit number operator
nq = sp * sm

# local cavity/vibration operators
am = destroy(Nf)
ap = am.dag()
# local cavity number operator, also works for local vibrations
nc = ap * am

# identities
Iq = qeye(2)
Ic = qeye(Nf)
Iv = qeye(Nf)

# frequencies
omega_q = LiH_params["w_q1"]
omega_c = LiH_params["w_cav"]
omega_v = LiH_params["w_vib1"]

# qubit transition dipole moment
mu_eg = LiH_params["qubit_1_dipole_moments"]["mu_eg"]

# field-squared dipole moment
d_eg = LiH_params["lambda_1"] * mu_eg

# qubit-cavity coupling 
g = -np.sqrt(omega_c / 2 ) * d_eg

# Make a list of S parameters for each qubit ranging from 0 to 10 in incremenets of 0.5
S_values = np.arange(0, 10.5, 0.1)

for Sv in S_values:
    
    # qubit-vib coupling strength
    lambda_v = omega_v * np.sqrt(Sv)

    # local qubit 1 Hamiltonian
    H_q1_local = -omega_q / 2 * sz

    # local qubit 2 Hamiltonian
    H_q2_local = -omega_q / 2 * sz

    # local qubit 1 vibration
    H_v1_local = omega_v * nc

    # local qubit 2 vibration
    H_v2_local = omega_v * nc

    # local cavity Hamiltonian
    H_cav_local = omega_q * nc

    # bare Hamiltonians in composite space
    H_q1 = tensor(H_q1_local, Iq, Iv, Iv, Ic)
    H_q2 = tensor(Iq, H_q2_local, Iv, Iv, Ic)

    H_v1 = tensor(Iq, Iq, H_v1_local, Iv, Ic)
    H_v2 = tensor(Iq, Iq, Iv, H_v2_local, Ic)

    H_cav = tensor(Iq, Iq, Iv, Iv, H_cav_local)

    # define H_bare_total as the sum of the three bare Hamiltonians for qubit 1, qubit 2, and cavity
    # all on the composite space
    H_bare_total = H_q1 + H_q2 + H_v1 + H_v2 + H_cav 

    # qubit1 - cavity interaction in order q1 x q2 x cav
    H_q1_cav = g * tensor(sp, Iq, Iv, Iv, am) + g * tensor(sm, Iq, Iv, Iv, ap)
    H_q2_cav = g * tensor(Iq, sp, Iv, Iv, am) + g * tensor(Iq, sm, Iv, Iv, ap)

    # qubit1 - vibration interaction in order q1 x q2 x vib1 x vib2 x cav
    H_q1_vib1 = lambda_v * tensor(sp * sm, Iq, (am + ap), Iv, Ic)
    H_q2_vib2 = lambda_v * tensor(Iq, sp * sm, Iv, (am + ap), Ic)


    # H_bare_total + vibronic coupling
    H_bare_vc = H_bare_total + H_q1_vib1 + H_q2_vib2

    tlist = np.linspace(0, 1400, 5000)

    # resonant SQRT iSWAP gate
    T0_1 = 20
    T_gate_1 = (1*pi)/(4 * np.abs(g))

    # resonant iSWAP gate
    T0_2 = T0_1 + T_gate_1
    T_gate_2 = (2 *pi)/(4 * np.abs(g))

    # time-dependent H; H_q1 + H_q2 + H_cav on at all times, H_q1_cav on until T1, H_q2_cav on from T1 to T1+T2
    H_t = [[H_bare_vc, wc_t], [H_q1_cav, w1_t], [H_q2_cav, w2_t]]

    # operators in composite space for expectation values
    nq1  = tensor(nq, Iq, Iv, Iv, Ic)
    nq2  = tensor(Iq, nq, Iv, Iv, Ic)
    ncav = tensor(Iq, Iq, Iv, Iv, nc)

    psi0 = tensor( basis(2,1), basis(2,0),basis(Nf,0), basis(Nf,0), basis(Nf,0))

    res = mesolve(H_t, psi0, tlist, [], e_ops=[])

    rho_final = res.states[-1]

    rho_qubits = ptrace(rho_final, [0,1])

    rho_qubits_ideal = ket2dm(tensor(phasegate(0), phasegate(pi/2)) * sqrtiswap() * tensor(basis(2,1), basis(2,0)))

    fide = fidelity(rho_qubits, rho_qubits_ideal)
    conc = concurrence(rho_qubits)

    # print S and fidelity and concurrence on one line
    print(f" {Sv:12.4f}, {fide:12.10f}, {conc:12.10f}")

