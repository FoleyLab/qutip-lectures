import numpy as np
from qutip import *
from qutip.core.gates import *

def jc_gate_calc(params):

    lam = params.get('lambda')
    g = params.get('g')
    T1 = params.get('T1')
    T2 = params.get('T2')

    # optional
    v = params.get('vib', 0)
    sqr_s = np.sqrt(params.get('S', 0))              # sqrt Hunag-Rhys
    cav_states = params.get('cav_states', 2)
    vib_states = params.get('vib_states', 0)
    w = params.get('w', 0.12086)                     # frequency of LiH in a.u.
    mu_eg = params.get('mu_eg', 1.0338263686725813)  # transition dipole moment of LiH in a.u.

    # resonant SQRT iSWAP gate
    T0_1 = 20
    T_gate_1 = T1

    # resonant iSWAP gate
    T0_2 = T0_1 + T_gate_1
    T_gate_2 = T2

    tlist = np.linspace(0, T0_2 + T2 + 100, int(T0_2 + T2 + 100) * 5)

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

    # operators for qubit sub systems
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sm = destroy(2)             # qubit lowering operator
    sp = sm.dag()               # qubit raising operator
    if vib_states > 0:
        a = destroy(vib_states) # vibrational lowering operator
        ap = a.dag()

    # cavity operators
    b = destroy(cav_states)
    bp = b.dag()

    # identity operators
    Iq = qeye(2)                # qubit identity
    Ic = qeye(cav_states)       # cavity idebtity

    # individual qubit and cavity operators
    H_q_i = - w / 2 * sz        # qubit hamiltonian
    H_cav_i = w * (bp * b )     # cavity hamiltonian

    if vib_states > 0:
        Iv = qeye(vib_states)   # vibration identity
        H_v_i = v * (ap * a)    # vibration hamiltonian

    if vib_states == 0:
        
        # qubit operators
        Hq_1 = tensor(H_q_i, Iq, Ic)
        Hq_2 = tensor(Iq, H_q_i, Ic)

        H_qubit = Hq_1 + Hq_2

        # cavity operaotr
        H_cav = tensor(Iq, Iq, H_cav_i)

        # qubit-cavity coupling
        _t1a = tensor(g * sp, Iq, b)
        _t1b = tensor(g * sm, Iq, bp)
        _t2a = tensor(Iq, g * sp, b)
        _t2b = tensor(Iq, g * sm, bp)

        H_q1_cav = _t1a + _t1b
        H_q2_cav = _t2a + _t2b

        # build full Hamiltonian and modulated Hamiltonian
        H_total = H_cav + H_qubit

        # time-dependent H; H_q1 + H_q2 + H_cav on at all times, H_q1_cav on until T1, H_q2_cav on from T1 to T1+T2
        H_t = [[H_total, wc_t], [H_q1_cav, w1_t], [H_q2_cav, w2_t]]

        psi0 = tensor( basis(2,1), basis(2,0), basis(cav_states,0))

        res = mesolve(H_t, psi0, tlist, [], e_ops=[])
        rho_final = res.states[-1]
        rho_qubits = ptrace(rho_final, [0,1])
        rho_qubits_ideal1 = ket2dm(tensor(phasegate(0), phasegate(np.pi/2)) * sqrtiswap() * tensor(basis(2,1), basis(2,0)))
        rho_qubits_ideal2 = ket2dm(tensor(phasegate(0), phasegate(-np.pi/2)) * sqrtiswap() * tensor(basis(2,1), basis(2,0)))

        f1 = fidelity(rho_qubits, rho_qubits_ideal1)
        f2 = fidelity(rho_qubits, rho_qubits_ideal2)
        conc = concurrence(rho_qubits)

        return f1, f2, conc
    else:
        # use when vibrations are included
        # qubit operators
        Hq_1 = tensor(H_q_i, Iq, Iv, Iv, Ic)
        Hq_2 = tensor(Iq, H_q_i, Iv, Iv,  Ic)

        H_qubit = Hq_1 + Hq_2

        # cavity operaotr
        H_cav = tensor(Iq, Iq, Iv, Iv, H_cav_i)

        # vibration operators
        Hv_1 = tensor(Iq, Iq, H_v_i, Iv, Ic)
        Hv_2 = tensor(Iq, Iq, Iv, H_v_i, Ic)

        H_vib = Hv_1 + Hv_2

        # qubit-vibrational coupling
        _qubit_excitation = sp * sm
        _vib_excitation = ap + a

        H_qv_1 = - sqr_s * v * tensor(_qubit_excitation, Iq, _vib_excitation, Iv, Ic)
        H_qv_2 = - sqr_s * v * tensor(Iq, _qubit_excitation, Iv, _vib_excitation, Ic)

        H_qv = H_qv_1 + H_qv_2

        # qubit-cavity coupling
        _t1a = tensor(g * sp, Iq, Iv, Iv, b)
        _t1b = tensor(g * sm, Iq, Iv, Iv, bp)
        _t2a = tensor(Iq, g * sp, Iv, Iv, b)
        _t2b = tensor(Iq, g * sm, Iv, Iv, bp)

        H_q1_cav = _t1a + _t1b
        H_q2_cav = _t2a + _t2b

        # build full Hamiltonian and modulated Hamiltonian
        H_total = H_cav + H_qubit + H_vib + H_qv

        # time-dependent H; H_q1 + H_q2 + H_cav on at all times, H_q1_cav on until T1, H_q2_cav on from T1 to T1+T2
        H_t = [[H_total, wc_t], [H_q1_cav, w1_t], [H_q2_cav, w2_t]]

        psi0 = tensor( basis(2,1), basis(2,0), basis(vib_states, 0), basis(vib_states, 0), basis(cav_states,0))

        res = mesolve(H_t, psi0, tlist, [], e_ops=[])
        rho_final = res.states[-1]
        rho_qubits = ptrace(rho_final, [0,1])
        rho_qubits_ideal1 = ket2dm(tensor(phasegate(0), phasegate(np.pi/2)) * sqrtiswap() * tensor(basis(2,1), basis(2,0)))
        rho_qubits_ideal2 = ket2dm(tensor(phasegate(0), phasegate(-np.pi/2)) * sqrtiswap() * tensor(basis(2,1), basis(2,0)))

        f1 = fidelity(rho_qubits, rho_qubits_ideal1)
        f2 = fidelity(rho_qubits, rho_qubits_ideal2)
        conc = concurrence(rho_qubits)

        return f1, f2, conc
