import numpy as np
from qutip import *

class MultiPartiteSystem:
    def __init__(self,
                 w_q1, w_q2,               # qubit frequencies
                 w_ho1, w_ho2, w_hoc,      # oscillator frequencies
                 g_q1_ho1, g_q2_ho2,       # individual couplings
                 g_q1_hoc, g_q2_hoc,       # common oscillator couplings
                 N_ho=10):                 # oscillator truncation level
        self.N_ho = N_ho

        # Operators for each subsystem
        self.sx = sigmax()
        self.sy = sigmay()
        self.sz = sigmaz()
        self.sm = destroy(2)   # qubit lowering operator

        self.a = destroy(N_ho) # oscillator lowering operator

        # Create identities for each space
        Iq = qeye(2)
        Io = qeye(N_ho)

        # Tensor structure:
        # [Q1, Q2, HO1, HO2, HOc]
        self.H0 = (
            0.5 * w_q1 * tensor(self.sz, Iq, Io, Io, Io) +
            0.5 * w_q2 * tensor(Iq, self.sz, Io, Io, Io) +
            w_ho1 * tensor(Iq, Iq, self.a.dag() * self.a, Io, Io) +
            w_ho2 * tensor(Iq, Iq, Io, self.a.dag() * self.a, Io) +
            w_hoc * tensor(Iq, Iq, Io, Io, self.a.dag() * self.a)
        )

        # Coupling terms
        self.H_coupling = (
            g_q1_ho1 * (tensor(self.sm + self.sm.dag(), Iq, self.a + self.a.dag(), Io, Io)) +
            g_q2_ho2 * (tensor(Iq, self.sm + self.sm.dag(), Io, self.a + self.a.dag(), Io)) +
            g_q1_hoc * (tensor(self.sm + self.sm.dag(), Iq, Io, Io, self.a + self.a.dag())) +
            g_q2_hoc * (tensor(Iq, self.sm + self.sm.dag(), Io, Io, self.a + self.a.dag()))
        )

        self.H_total = self.H0 + self.H_coupling

    def get_hamiltonian(self):
        return self.H_total

    def dimension(self):
        return self.H_total.shape


