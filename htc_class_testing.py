import numpy as np
from qutip import *

class HolsteinTavisCummings:
    def __init__(self, params):
        # Required parameters
        self.w_q1 = params['w_q1']
        self.w_q2 = params['w_q2']
        self.w_vib1 = params['w_vib1']
        self.w_vib2 = params['w_vib2']
        self.w_cav = params['w_cav']
        self.S_1 = params['S_1']
        self.S_2 = params['S_2']
        self.lambda_1 = params['lambda_1']
        self.lambda_2 = params['lambda_2']

        # Optional parameters with defaults
        self.N_vib = params.get('N_vib', 10)
        self.N_cav = params.get('N_cav', 10)
        self.qubit_1_dipole_moments = params.get('qubit_1_dipole_moments', {'mu_g': 0, 'mu_e': 0, 'mu_eg': 1})
        self.qubit_2_dipole_moments = params.get('qubit_2_dipole_moments', {'mu_g': 0, 'mu_e': 0, 'mu_eg': 1})
        self.pauli_fierz = params.get('pauli_fierz', False)
        self.dse_term = params.get('dse_term', False)
        self.simple_g = params.get('simple_g', False)
        self.g1 = params.get('g1', 0.1)  # coupling strength for qubit 1
        self.g2 = params.get('g2', 0.1)  # coupling
        self.sigma_z_Hq = params.get('sigma_z_Hq', False)  # whether to formulate qubit Hamiltonian using sigma_z operator or sigma^+ sigma^-
        self.cavity_zero_point = params.get('cavity_zero_point', 0.0)  # zero-point energy of the cavity mode
        self.vibrational_zero_point = params.get('vibrational_zero_point', 0.0)  # zero-point energy of the vibrational modes
        self.qubit_zero_point = params.get('qubit_zero_point', 0.0)  # zero-point energy of the qubits
        self.include_cavity = params.get('include_cavity', False)  # whether to include the cavity in polaron basis


        
        # Operators for qubit sub systems
        self.sx = sigmax()
        self.sy = sigmay()
        self.sz = sigmaz()
        self.sm = destroy(2) # qubit lowering operator
        self.sp = self.sm.dag() # qubit raising operator

        # Operators for vibrational modes
        # only build if self.N_vib > 1
        if self.N_vib > 1:
            self.a = destroy(self.N_vib) # vibrational lowering operator
            self.ap = self.a.dag() # vibrational raising operator
            self.Ivib = qeye(self.N_vib)  # identity operator for vibrational modes


        # Operators for cavity mode
        self.b = destroy(self.N_cav)
        self.bp = self.b.dag()

        # Create identities for each space
        self.Iq = qeye(2)
        self.Icav = qeye(self.N_cav)


        # Note on Tensor structure:
        # [Q1, Q2, V1, V2, Cav]

        # let's build base operators for the full Hilbert space
        if self.N_vib > 1:  # full Hilbert space includes vibrational
            self.sigmaz_1 = tensor(self.sz, self.Iq, self.Ivib, self.Ivib, self.Icav)
            self.sigmaz_2 = tensor(self.Iq, self.sz, self.Ivib, self.Ivib, self.Icav)
            self.number_cavity = tensor(self.Iq, self.Iq, self.Ivib, self.Ivib, self.b.dag() * self.b)

        else:  # full Hilbert space does not include vibrational modes
            self.sigmaz_1 = tensor(self.sz, self.Iq, self.Icav)
            self.sigmaz_2 = tensor(self.Iq, self.sz, self.Icav)
            self.number_cavity = tensor(self.Iq, self.Iq, self.b.dag() * self.b)

    def build_qubit_dipole_operator(self):
        """ method to build the dipole operator on a single qubit subspace.

        Args:
            mu_g (float): ground state dipole moment
            mu_e (float): excited state dipole moment
            mu_eg (float): transition dipole moment

        Returns:
            _mu (Qobj): dipole operator for the qubit subspace
        """

        # Define the dipole operator for a single qubit subspace
        sig_p = self.sp
        sig_m = self.sm

        self.qubit_1_dipole_operator = self.qubit_1_dipole_moments['mu_g'] * sig_p * sig_m
        self.qubit_1_dipole_operator += self.qubit_1_dipole_moments['mu_e'] * sig_m * sig_p
        self.qubit_1_dipole_operator += self.qubit_1_dipole_moments['mu_eg'] * (sig_p + sig_m)

        self.qubit_2_dipole_operator = self.qubit_2_dipole_moments['mu_g'] * sig_p * sig_m
        self.qubit_2_dipole_operator += self.qubit_2_dipole_moments['mu_e'] * sig_m * sig_p
        self.qubit_2_dipole_operator += self.qubit_2_dipole_moments['mu_eg'] * (sig_p + sig_m)

        # Build the full dipole operator on the full Hilbert space
        if self.N_vib > 1: # full hilbert space includes vibrational modes
            self.dipole_operator_q1 = tensor(self.qubit_1_dipole_operator, self.Iq, self.Ivib, self.Ivib, self.Icav)
            self.dipole_operator_q2 = tensor(self.Iq, self.qubit_2_dipole_operator, self.Ivib, self.Ivib, self.Icav)

        else: # full hilbert space does not include vibrational modes
            self.dipole_operator_q1 = tensor(self.qubit_1_dipole_operator, self.Iq,  self.Icav)
            self.dipole_operator_q2 = tensor(self.Iq, self.qubit_2_dipole_operator, self.Icav)

        self.dipole_operator = self.dipole_operator_q1 + self.dipole_operator_q2

    def build_qubit_hamiltonians(self):
        """" Method to build the qubit 1 and qubit 2 Hamiltonians on the full Hilbert space."""
        if self.sigma_z_Hq:
            # Use sigma_z operator to build the qubit Hamiltonian
            self.H_q1_individual = - self.w_q1 / 2 * self.sz
            self.H_q2_individual = - self.w_q2 / 2 * self.sz
        else:
            # Use raising and lowering operators to build the qubit Hamiltonian
            self.H_q1_individual = self.w_q1 * (self.sp * self.sm + self.qubit_zero_point)
            self.H_q2_individual = self.w_q2 * (self.sp * self.sm + self.qubit_zero_point)

        if self.N_vib > 1:  # full Hilbert space includes vibrational modes
            self.H_q1 = tensor(self.H_q1_individual, self.Iq, self.Ivib, self.Ivib, self.Icav)
            self.H_q2 = tensor(self.Iq, self.H_q2_individual, self.Ivib, self.Ivib, self.Icav)
        else:  # full Hilbert space does not include vibrational modes
            self.H_q1 = tensor(self.H_q1_individual, self.Iq, self.Icav)
            self.H_q2 = tensor(self.Iq, self.H_q2_individual, self.Icav)

        self.H_qubit = self.H_q1 + self.H_q2

    def build_vibrational_hamiltonians(self):
        """ Method to build the vibrational Hamiltonians for both modes on the full Hilbert space."""
        if self.N_vib > 1:
            self.H_vib1_individual = self.w_vib1 * (self.ap * self.a + self.vibrational_zero_point)
            self.H_vib2_individual = self.w_vib2 * (self.ap * self.a + self.vibrational_zero_point)
            self.H_vib1 = tensor(self.Iq, self.Iq, self.H_vib1_individual, self.Ivib, self.Icav)
            self.H_vib2 = tensor(self.Iq, self.Iq, self.Ivib, self.H_vib2_individual, self.Icav)
            self.H_vibrational = self.H_vib1 + self.H_vib2
        else:
            self.H_vibrational = 0 

    def build_cavity_hamiltonian(self):
        """ Method to build the cavity Hamiltonian on the full Hilbert space."""
        self.H_cav_individual = self.w_cav * (self.bp * self.b + self.cavity_zero_point)
        if self.N_vib > 1: # full Hilbert space includes vibrational modes
            self.H_cav = tensor(self.Iq, self.Iq, self.Ivib, self.Ivib, self.H_cav_individual)
        else: # full Hilbert space does not include vibrational modes
            self.H_cav = tensor(self.Iq, self.Iq, self.H_cav_individual)

    def build_qubit_vibrational_coupling(self):
        """ Method to build the qubit-vibrational coupling Hamiltonians on the full Hilbert space."""
        if self.N_vib > 1:
            # first build the individual coupling terms
            _qubit_excitation = self.sp * self.sm
            _vib_excitation = self.ap + self.a

            # qubit 1 coupling constant is squared Huang-Rhys factor
            _lambda_1 = np.sqrt(self.S_1) 
            _lambda_2 = np.sqrt(self.S_2)

            # now scale by the Huang-Rhys factors and build on the full Hilbert space
            self.H_q1_vib1 = - _lambda_1 * self.w_vib1 * tensor(_qubit_excitation, self.Iq, _vib_excitation, self.Ivib, self.Icav)

            self.H_q2_vib2 = - _lambda_2 * self.w_vib2 * tensor(self.Iq, _qubit_excitation, self.Ivib, _vib_excitation, self.Icav)
            #Combine the two coupling terms
            self.H_qubit_vibrational_coupling = self.H_q1_vib1 + self.H_q2_vib2

        else:
            # If N_vib is 1, no vibrational modes, so no coupling
            self.H_qubit_vibrational_coupling = 0 

    def build_qubit_cavity_coupling(self, pauli_fierz=False, dse_term=False, simple_g = False):
        """ Method to build the qubit-cavity bilinear coupling Hamiltonians on the full Hilbert space.

        Args:
            pauli_fierz (bool): If True, use the Pauli-Fierz form of the coupling.

            if False, use Tavis-Cummings form of the coupling.
            Default is False.
        """
        if pauli_fierz:
            # Pauli-Fierz coupling - needs the dipole operator
            if not hasattr(self, 'dipole_operator'):
                self.build_qubit_dipole_operator()

            _cavity_excitation = self.b + self.bp
            _qubit_1_d_operator = np.sqrt(self.w_q1/2) * self.lambda_1 * self.dipole_operator_q1
            _qubit_2_d_operator = np.sqrt(self.w_q2/2) * self.lambda_2 * self.dipole_operator_q2

            if self.N_vib > 1:  # full Hilbert space includes vibrational modes
                self.H_q1_cav = tensor(_qubit_1_d_operator, self.Iq, self.Ivib, self.Ivib, _cavity_excitation)
                self.H_q2_cav = tensor(self.Iq, _qubit_2_d_operator, self.Ivib, self.Ivib, _cavity_excitation)

            else:  # full Hilbert space does not include vibrational modes
                self.H_q1_cav = tensor(_qubit_1_d_operator, self.Iq, _cavity_excitation)
                self.H_q2_cav = tensor(self.Iq, _qubit_2_d_operator, _cavity_excitation)

            self.H_qubit_cavity_coupling = self.H_q1_cav + self.H_q2_cav

            if dse_term:
                _d1 = self.lambda_1 * self.dipole_operator_q1
                _d2 = self.lambda_2 * self.dipole_operator_q2

                if self.N_vib > 1:  # full Hilbert space includes vibrational modes
                    d1 = tensor(_d1, self.Iq, self.Ivib, self.Ivib, self.Icav)
                    d2 = tensor(self.Iq, _d2, self.Ivib, self.Ivib, self.Icav)
                else:  # full Hilbert space does not include vibrational modes
                    d1 = tensor(_d1, self.Iq, self.Icav)
                    d2 = tensor(self.Iq, _d2, self.Icav)

                self.H_dipole_self_energy = 0.5 * (d1 + d2) @ (d1 + d2)

        else:
            print("Using Tavis-Cummings coupling")
            # Tavis-Cummings coupling - only TDM element needed
            if not hasattr(self, 'qubit_1_dipole_operator'):
                self.build_qubit_dipole_operator()
            if simple_g:
                print("Using simple g coupling")
                self._g1 = self.g1
                self._g2 = self.g2
                print(f"g1: {self.g1}, g2: {self.g2}")
            else:
                print("Using g coupling with TDM")
                self._g1 = np.sqrt(self.w_q1/2) * self.lambda_1 * self.qubit_1_dipole_operator[0,1]
                self._g2 = np.sqrt(self.w_q2/2) * self.lambda_2 * self.qubit_2_dipole_operator[0,1]

            # Build the coupling terms on the full Hilbert space
            if self.N_vib > 1:  # full Hilbert space includes vibrational modes
                _t1a = tensor(self._g1 * self.sp, self.Iq, self.Ivib, self.Ivib, self.b)
                _t1b = tensor(self._g1 * self.sm, self.Iq, self.Ivib, self.Ivib, self.bp)
                _t2a = tensor(self.Iq, self._g2 * self.sp, self.Ivib, self.Ivib, self.b)
                _t2b = tensor(self.Iq, self._g2 * self.sm, self.Ivib, self.Ivib, self.bp)
            else:  # full Hilbert space does not include vibrational modes
                _t1a = tensor(self._g1 * self.sp, self.Iq, self.b)
                _t1b = tensor(self._g1 * self.sm, self.Iq, self.bp)
                _t2a = tensor(self.Iq, self._g2 * self.sp, self.b)
                _t2b = tensor(self.Iq, self._g2 * self.sm, self.bp)

            self.H_q1_cav = _t1a + _t1b
            self.H_q2_cav = _t2a + _t2b
            self.H_qubit_cavity_coupling = self.H_q1_cav + self.H_q2_cav

    def build_polaron_basis(self):
        """ Method to build the polaron basis for the system."""
        #Build Hamitonian with qubit and vibrations, exclude cavity, get eigenvalues
        # self.H_pt = self.H_qubit + self.H_vibrational + self.H_qubit_vibrational_coupling 
        # self.H_pt = self.H_qubit + self.H_vibrational + self.H_qubit_vibrational_coupling + self.H_cav
        # self.vals, self.vecs = self.H_pt.eigenstates()

        if self.include_cavity:
            self.H_pt = self.H_qubit + self.H_vibrational + self.H_qubit_vibrational_coupling + self.H_cav
            self.vals, self.vecs = self.H_pt.eigenstates()
            self.H_pt += self.H_qubit_cavity_coupling
        
        else:
            self.H_pt = self.H_qubit + self.H_vibrational + self.H_qubit_vibrational_coupling
            self.vals, self.vecs = self.H_pt.eigenstates()
            #add cavity Hamiltonian and qubit-cavity coupling to Hamiltonian
            self.H_pt += self.H_qubit_cavity_coupling + self.H_cav

        #Build matrix from qubit eigenvectors
        self.vecs_new = [vec.full() for vec in self.vecs]
        self.H_vecs = Qobj(np.hstack(self.vecs_new),dims=self.H_pt.dims)

        #Perform polaron transformation
        self.H_p = self.H_vecs.dag() * self.H_pt * self.H_vecs
        
    @staticmethod
    def step_t(w1, w2, t0, t):
        """
        Step function that goes from w1 to w2 at time t0
        as a function of t. 
        """
        return w1 + (w2 - w1) * (t > t0)

    def w1_t(self, t, T0_1, T_gate_1):
        return self.w_q1 + self.step_t(0.0, self.w_cav-self.w_q1, T0_1, t) - self.step_t(0.0, self.w_cav-self.w_q1, T0_1+T_gate_1, t)
    
    def w2_t(self, t, T0_2, T_gate_2):
        return self.w_q2 + self.step_t(0.0, self.w_cav-self.w_q2, T0_2, t) - self.step_t(0.0, self.w_cav-self.w_q2, T0_2+T_gate_2, t)

    def build_polaron_basis_t(self, t, t1, t_gate_1, t2, t_gate_2, return_vectors=False):
        """Method to build the polaron basis for the system at a given time t.

        Args:
            t (float): Current time in the simulation.
            t1 (float): Time at which the first qubit frequency changes.
            t_gate_1 (float): Duration of the first qubit frequency change.
            t2 (float): Time at which the second qubit frequency changes.
            t_gate_2 (float): Duration of the second qubit frequency change.
            return_vectors (bool): If True, return the eigenvectors of the polaron Hamiltonian.
        Returns:
            H_polaron (Qobj): The polaron Hamiltonian at time t.
            if return_vectors is True:
                H_vectors (Qobj): The eigenvectors of the polaron Hamiltonian.
        """

        H1 = -0.5 * self.w1_t(t, t1, t_gate_1) * self.sigmaz_1
        H2 = -0.5 * self.w2_t(t, t2, t_gate_2) * self.sigmaz_2

        #build polaron transformation Hamiltonian
        #self.H_change = H1 + H2 + self.H_vibrational + self.H_qubit_vibrational_coupling #+ self.H_cav
        #self.values, self.vectors = self.H_change.eigenstates()

        #add cavity Hamiltonian and qubit-cavity coupling to Hamiltonian
        #self.H_change += self.H_qubit_cavity_coupling + self.H_cav

        self.H_change = H1 + H2 + self.H_vibrational + self.H_qubit_vibrational_coupling + self.H_cav + self.H_qubit_cavity_coupling

        #Build matrix from qubit eigenvectors
        #self.vectors_new = [vec.full() for vec in self.vectors]
        #self.H_vectors = Qobj(np.hstack(self.vectors_new),dims=self.H_change.dims)

        #Perform polaron transformation
        #self.H_polaron = self.H_vectors.dag() * self.H_change * self.H_vectors
        self.H_polaron = self.H_vecs.dag() * self.H_change * self.H_vecs

        if return_vectors:
            return self.H_polaron, self.H_vecs
        else:
            return self.H_polaron
        
    def build_time_dependent_bare_Hamiltonian(self, t, t1, t_gate_1, t2, t_gate_2):
        """Method to build the time-dependent bare Hamiltonian for the system at a given time t.

        Args:
            t (float): Current time in the simulation.
            t1 (float): Time at which the first qubit frequency changes.
            t_gate_1 (float): Duration of the first qubit frequency change.
            t2 (float): Time at which the second qubit frequency changes.
            t_gate_2 (float): Duration of the second qubit frequency change.
            return_vectors (bool): If True, return the eigenvectors of the polaron Hamiltonian.
        Returns:
            H_bare (Qobj): The bare Hamiltonian at time t.
        """
        H1 = -0.5 * self.w1_t(t, t1, t_gate_1) * self.sigmaz_1
        H2 = -0.5 * self.w2_t(t, t2, t_gate_2) * self.sigmaz_2

        self.H_bare = H1 + H2 + self.H_vibrational + self.H_qubit_vibrational_coupling + self.H_qubit_cavity_coupling + self.H_cav

        return self.H_bare
    
    def build_hamiltonian(self):
        self.build_qubit_hamiltonians()
        self.build_cavity_hamiltonian()
        self.build_qubit_cavity_coupling(self.pauli_fierz, self.dse_term, self.simple_g)
        # Combine all parts to form the total Hamiltonian
        if self.N_vib > 1:
            self.build_vibrational_hamiltonians()
            self.build_qubit_vibrational_coupling()
            self.H_total = self.H_qubit + self.H_vibrational + self.H_cav + self.H_qubit_vibrational_coupling + self.H_qubit_cavity_coupling
        else: # the vibrational terms are zero anyway, but also fine to just not add them!
            self.H_total = self.H_qubit + self.H_cav + self.H_qubit_cavity_coupling

        self.eigenvalues, self.eigenstates = self.H_total.eigenstates()
        self.build_polaron_basis()






    

