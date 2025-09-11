import numpy as np
from itertools import product

class HTC_Numpy:
    def __init__(self, params): 
        """
        Parameters
        ----------
        params : dict
            Dictionary containing simulation parameters. Expected keys:
            
            # General Hilbert space truncations
            dim_cavity : int
                Truncation dimension of cavity bosonic mode, default: 2
            dim_vib : int
                Truncation dimension of vibrational bosonic modes, default: 2

            n_vib_per_cavity : int
                Number of vibrational modes on the cavity, default: 1

            # For simplicity, we assume all qubits have the same parameters
            n_qubits : int
                Number of qubits, default: 2
            n_vib_per_qubit : int
                Number of vibrational modes per qubit, default: 1

            # Uncoupled basis parameters
            cavity_lambda : float
                cavity-qubit coupling strength (λ), default: 0.1

            qubit_huang_rhys : float
                Huang-Rhys factor for qubit vibrations (S), default: 1.0

            qubit_dipole_values : list of float
                Dipole matrix elements for qubit: [μ_gg, μ_eg, μ_ee], default: [1.0, 1.0, 1.0]
            
            freq_cavity : float
                Frequency of the cavity mode (ω_c), default: 1.0

            freq_qubit : float
                Frequency of the qubit transition (ω_q), default: 1.0

            freq_vib_qubit : float
                Frequency of vibrational modes on each qubit (ω_v,q), default: 0.1

            freq_vib_cavity : float
                Frequency of vibrational modes on the cavity (ω_v,c); default: 0.

            # Polaritonic basis parameters
            polariton_energies : list of float
                Energies of polariton states.  default: [0, 0.9, 1.1, 2.0]

            polariton_vib_frequencies : list of float
                Vibrational frequencies on each polariton surface, default: [0.1, 0.1, 0.1, 0.1]

            polariton_huang_rhys : list of float
                Huang-Rhys factors for vibronic coupling between polaritons, default: [0.05, 0.05, 0.05]
                Corresponds to couplings between (pol1-pol2), (pol1-pol3), (pol1-pol4)

            

        """

        # --- Hilbert space dimensions ---
        self.dim_qubit = 2  # Assuming qubits are two-level systems
        self.dim_cavity = params.get('dim_cavity', 2)
        self.dim_vib = params.get('dim_vib', 2)
        self.n_qubits = params.get('n_qubits', 2)
        self.n_vib_per_qubit = params.get('n_vib_per_qubit', 1)
        self.n_vib_per_cavity = params.get('n_vib_per_cavity', 1)

        # create identities on each individual subspace
        self.id_qubit = np.eye(self.dim_qubit)
        self.id_cavity = np.eye(self.dim_cavity)
        self.id_vib = np.eye(self.dim_vib)

        # --- Uncoupled basis parameters (incomplete) ---
        self.freq_cavity = params.get('freq_cavity', 1.0)
        self.cavity_lambda = params.get('cavity_lambda', 0.1)  # Cavity displacement
        self.freq_qubit = params.get('freq_qubit', 1.0)
        self.freq_vib_qubit = params.get('freq_vib_qubit', 0.1)
        self.freq_vib_cavity = params.get('freq_vib_cavity', 0.1)
        self.qubit_huang_rhys = params.get('qubit_huang_rhys', 1.0)  # Huang-Rhys factor S
        self.qubit_dipole_values = params.get('qubit_dipole_values', [1.0, 1.0, 1.0]) # [μ_gg, μ_eg, μ_ee]

        # Qubit-vibration coupling strength λ_q = sqrt(S) * ω_v
        self.qubit_vib_coupling = np.sqrt(self.qubit_huang_rhys) * self.freq_vib_qubit

        # dipole matrix for qubit
        self.qubit_dipole_matrix = np.array([[self.qubit_dipole_values[0], self.qubit_dipole_values[1]],
                                             [self.qubit_dipole_values[1], self.qubit_dipole_values[2]]])
        
        # d matrix for qubit-cavity coupling
        self.qubit_d_matrix = self.cavity_lambda * self.qubit_dipole_matrix
        
        # --- Polaritonic basis parameters ---
        self.polariton_energies = params.get(
            'polariton_energies', [0, 0.9, 1.1, 2.0]
        )
        (
            self.energy_pol1,
            self.energy_pol2,
            self.energy_pol3,
            self.energy_pol4
        ) = self.polariton_energies

        self.polariton_vib_frequencies = params.get(
            'polariton_vib_frequencies',
            [self.freq_vib_qubit] * 4
        )
        (
            self.freq_vib_pol1,
            self.freq_vib_pol2,
            self.freq_vib_pol3,
            self.freq_vib_pol4
        ) = self.polariton_vib_frequencies

        self.polariton_huang_rhys = params.get(
            'polariton_huang_rhys',
            [0.05, 0.05, 0.05]
        )
        # get the coupling strenghts between state i and j as λ_ij = sqrt(S_ij) * ω_i
        self.polariton_vib_coupling_12 = np.sqrt(self.polariton_huang_rhys[0]) * self.freq_vib_pol1
        self.polariton_vib_coupling_13 = np.sqrt(self.polariton_huang_rhys[1]) * self.freq_vib_pol1
        self.polariton_vib_coupling_14 = np.sqrt(self.polariton_huang_rhys[2]) * self.freq_vib_pol1



        # ordering: [q1, c, q2, v1, v2] 
        self.sub_dims = [self.dim_qubit, self.dim_cavity, self.dim_qubit, self.dim_vib, self.dim_vib]
        self.dim = np.prod(self.sub_dims)
        print(f"Total Hilbert space dimension: {self.dim}")
        print(f"cavity coupling λ: {self.cavity_lambda}")
        print(f"qubit dipole matrix:\n{self.qubit_d_matrix}")
        print(f"Qubit frequency is {self.freq_qubit}, cavity frequency is {self.freq_cavity}, vib frequency is {self.freq_vib_qubit}")
        # Build basis labels
        self.labels = self._build_labels()

    # ------------------------------
    # Basis and labels for uncoupled basis based on order [q1, c, q2, v1, vc, v2]
    # ------------------------------
    def _build_labels(self):
        cav_labels   = [f"{n}" for n in range(self.dim_cavity)]
        qubit_labels = ["g", "e"] if self.dim_qubit == 2 else [str(i) for i in range(self.dim_qubit)]
        vib_labels   = [f"{n}" for n in range(self.dim_vib)]

        labels = []
        if self.n_qubits == 2 and self.n_vib_per_qubit * self.n_qubits == 2:
            # loop explicitly in the correct order
            for q1 in qubit_labels:
                for cav in cav_labels:
                    for q2 in qubit_labels:
                        for v1 in vib_labels:
                            for v2 in vib_labels:
                                label = f"|{q1}, {cav}, {q2}, {v1}, {v2}>"
                                labels.append(label)
        elif self.n_qubits == 1 and self.n_vib_per_qubit * self.n_qubits == 1:
            # loop explicitly in the correct order
            for q1 in qubit_labels:
                for cav in cav_labels:
                    for v1 in vib_labels:
                        label = f"|{q1}, {cav}, {v1}>"
                        labels.append(label)

        return labels

    def basis_state(self, dim, index):
        """Return computational basis vector for given index"""
        state = np.zeros((dim, 1))
        state[index, 0] = 1
        return state
    
    def build_projector(self, dim, index_a, index_b):
        """Return projector |a><b| for given basis indices"""
        state_a = self.basis_state(dim, index_a)
        state_b = self.basis_state(dim, index_b)
        return state_a @ state_b.T.conj()
    
    def build_polariton_hamiltonian(self):
        """
        Build the Hamiltonian in the polaritonic basis without vibrational modes.
        The basis states are |pol1>, |pol2>, |pol3>, |pol4>.
        """
        pol_dim = len(self.polariton_energies)
        H = self.energy_pol1 * self.build_projector(pol_dim, 0, 0)  # |pol1><pol1|
        H += self.energy_pol2 * self.build_projector(pol_dim, 1, 1)  # |pol2><pol2|
        H += self.energy_pol3 * self.build_projector(pol_dim, 2, 2)  # |pol3><pol3|
        H += self.energy_pol4 * self.build_projector(pol_dim, 3, 3)  # |pol4><pol4|

        self.H_polariton_local = np.copy(H)

        # embed into full Hilbert space: H ⊗ I_qubit ⊗ I_vib ⊗ I_vib
        self.H_polariton_composite = self._tensor(H, self.id_qubit, self.id_vib, self.id_vib)
        return self.H_polariton_composite  
    
    def build_polariton_vibrational_hamiltonian(self):
        """
        Build the Hamiltonian according to
        H_pol-vib = |pol1><pol1| ω_v,1 b†b
                  + |pol2><pol2| ω_v,2 b†b
                  + |pol3><pol3| ω_v,3 b†b
                  + |pol4><pol4| ω_v,4 b†b
        where b†b are the creation and annihilation operators for the vibrational mode of qubit 1.
        This needs to be embedded into the full Hilbert space.
        """
        # build b†b operator for vibrational mode
        b_dagger = self.creation(self.dim_vib)
        b = self.annihilation(self.dim_vib)
        n_vib = b_dagger @ b  # number operator

        pol_dim = len(self.polariton_energies)
        # |pol1><pol1| projector
        pol1_proj = self.build_projector(pol_dim, 0, 0)
        # |pol2><pol2| projector
        pol2_proj = self.build_projector(pol_dim, 1, 1)
        # |pol3><pol3| projector
        pol3_proj = self.build_projector(pol_dim, 2, 2)
        # |pol4><pol4| projector
        pol4_proj = self.build_projector(pol_dim, 3, 3)

        H  = self.freq_vib_pol1 * self._tensor(pol1_proj, self.id_qubit, n_vib, self.id_vib)
        H += self.freq_vib_pol2 * self._tensor(pol2_proj, self.id_qubit, n_vib, self.id_vib)
        H += self.freq_vib_pol3 * self._tensor(pol3_proj, self.id_qubit, n_vib, self.id_vib)
        H += self.freq_vib_pol4 * self._tensor(pol4_proj, self.id_qubit, n_vib, self.id_vib)

        self.H_polariton_vibrational = np.copy(H)
        return H
    
    def build_polariton_vib_coupling(self):
        """
        Build the vibronic coupling Hamiltonian in the polaritonic basis.
        H_coupling = λ_12 |pol1><pol2| x I_q2 x (b† + b) x I_v2
                   + λ_13 |pol1><pol3| x I_q2 x (b† + b) x I_v2
                   + λ_14 |pol1><pol4| x I_q2 x (b† + b) x I_v2   
                   + h.c.
        where λ_ij are the vibronic couplings between polaritonic states.
        """
        # build (b† + b) operator for vibrational mode
        b_dagger = self.creation(self.dim_vib)
        b = self.annihilation(self.dim_vib)
        _sub_vib_coupling_op = b_dagger + b
  

        # get polariton dimension for the projection operators
        pol_dim = len(self.polariton_energies)

        # build the coupling Hamiltonian 
        _proj_12 = self.build_projector(pol_dim, 0, 1)  # |pol1><pol2|
        _proj_13 = self.build_projector(pol_dim, 0, 2)  # |pol1><pol3|
        _proj_14 = self.build_projector(pol_dim, 0, 3)  # |pol1><pol4|

        H = self.polariton_vib_coupling_12 * self._tensor(_proj_12, self.id_qubit, _sub_vib_coupling_op, self.id_vib)
        H += self.polariton_vib_coupling_13 * self._tensor(_proj_13, self.id_qubit, _sub_vib_coupling_op, self.id_vib)    
        H += self.polariton_vib_coupling_14 * self._tensor(_proj_14, self.id_qubit, _sub_vib_coupling_op, self.id_vib)    
        # add hermitian conjugate
        H += self.polariton_vib_coupling_12 * self._tensor(_proj_12.T.conj(), self.id_qubit, _sub_vib_coupling_op, self.id_vib)   
        H += self.polariton_vib_coupling_13 * self._tensor(_proj_13.T.conj(), self.id_qubit, _sub_vib_coupling_op, self.id_vib)       
        H += self.polariton_vib_coupling_14 * self._tensor(_proj_14.T.conj(), self.id_qubit, _sub_vib_coupling_op, self.id_vib)   

        self.H_polariton_vib_coupling = np.copy(H)
        return H
    
    def build_polariton_transformation(self):
        """
        Build the transformation matrix from the uncoupled basis to the polaritonic basis.
        The transformation is defined by the polaritonic states in terms of the uncoupled states.
        For simplicity, we assume a fixed transformation here.
        """
        # build local qubit Hamiltonian first
        H_q1 = self.build_local_qubit_hamiltonian()

        # build local cavity Hamiltonian
        H_c = self.build_local_cavity_hamiltonian()

        # build b and b† operators for cavity mode
        b_dagger = self.creation(self.dim_cavity)
        b = self.annihilation(self.dim_cavity)

        # add bilinear coupling
        H_blc = -np.sqrt(self.freq_cavity / 2) * np.kron(self.qubit_d_matrix, (b_dagger + b))

        # add dipole self-energy term
        H_dse = 1 / 2 * np.kron(self.qubit_d_matrix @ self.qubit_d_matrix, np.eye(self.dim_cavity))

        # local H_PF
        self.H_PF_local = np.kron(H_q1, self.id_cavity) + np.kron(self.id_qubit, H_c) + H_blc + H_dse
        # embed into full Hilbert space: H ⊗ I_qubit ⊗ I_vib ⊗ I_vib
        self.H_PF_composite = self._tensor(self.H_PF_local, self.id_qubit, self.id_vib, self.id_vib)

        # diagonalize to generate U matrix on full Hilbert space
        energies, eigvecs = np.linalg.eigh(self.H_PF_composite)

        self.U_polariton = np.copy(eigvecs)

        # build matrix of diagonals from energies
        self.H_PF_Transformed = np.diag(energies)

        return self.U_polariton

    
    def build_local_qubit_hamiltonian(self):
        """
        Build the local Hamiltonian for the qubits in the uncoupled basis.
        H_qubit = (ω_q / 2) σ_z
        where σ_z is the Pauli Z operator for the qubit.
        """
        H = -(self.freq_qubit / 2) * self.sigma_z()

        self.H_qubit_local = np.copy(H)

        return H
    
    def build_local_cavity_hamiltonian(self):
        """
        Build the local Hamiltonian for the cavity in the uncoupled basis.
        H_cavity = ω_c a†a
        where a†a is the number operator for the cavity mode.
        """
        # build a†a operator for cavity mode
        a_dagger = self.creation(self.dim_cavity)
        a = self.annihilation(self.dim_cavity)
        n_cav = a_dagger @ a
        H = self.freq_cavity * n_cav
        self.H_cavity_local = np.copy(H)
        return H
    
    def build_qubit2_hamiltonian(self):
        """
        Build the Hamiltonian for the second qubit in the uncoupled basis.
        H_qubit2 = (ω_q / 2) σ_z
        where σ_z is the Pauli Z operator for the qubit.
        """
        H = -(self.freq_qubit / 2) * self.sigma_z()

        # embed into full Hilbert space: I_qubit ⊗ I_cavity ⊗ σ_z ⊗ I_vib ⊗ I_vib
        H = self._tensor(self.id_qubit, self.id_cavity, H, self.id_vib, self.id_vib)

        self.H_qubit2_composite = np.copy(H)
        return H
    
    def build_qubit2_vibrational_hamiltonian(self):
        """
        Build the vibrational Hamiltonian for the qubits in the uncoupled basis.
        H_vib = ω_v b†b
        where b†b is the number operator for the vibrational mode.
        """
        # build b†b operator for vibrational mode
        b_dagger = self.creation(self.dim_vib)
        b = self.annihilation(self.dim_vib)
        n_vib = b_dagger @ b  # number operator

        # embed into full Hilbert space: I_qubit ⊗ I_cavity ⊗ I_qubit ⊗ I_vib ⊗ b†b
        H = self.freq_vib_qubit * self._tensor(self.id_qubit, self.id_cavity, self.id_qubit, self.id_vib, n_vib)
        self.H_qubit2_vibrational = np.copy(H)
        return H
    
    def build_qubit2_vib_coupling(self):
        """
        Build the qubit-vibrational coupling Hamiltonian in the uncoupled basis.
        H_coupling = λ_q \sigma^+ \sigma^- (b† + b)
        where λ_q is the qubit-vibration coupling strength.
        """
        # build (b† + b) operator for vibrational mode
        b_dagger = self.creation(self.dim_vib)
        b = self.annihilation(self.dim_vib)
        vib_coupling_op = b_dagger + b

        # qubit raising and lowering operators
        sigma_plus = self.sigma_plus()
        sigma_minus = self.sigma_minus()
        qubit_proj = sigma_plus @ sigma_minus
        
        # build coupling term
        H = self.qubit_vib_coupling * self._tensor(self.id_qubit, self.id_cavity, qubit_proj, self.id_vib, vib_coupling_op)
        
        self.H_qubit2_vib_coupling = np.copy(H)
        return H


    def index_of(self, label):
        """Return basis index for given label string"""
        return self.labels.index(label)
    
    def represent_basis_in_eigenbasis(self, basis_labels, eigvecs, energies=None, tol=1e-6):
        """
        For each original basis state |cav, q1, v1, q2, v2>, print
        its expansion in the eigenbasis, |cav_j, q1_j, v1_j, q2_j, v2_j> = sum_i c_{j,i} |psi_i>,
        where c_{j,i} = <psi_i|cav_j, q1_j, v1_j, q2_j, v2_j> = conj(eigvecs[j,i)

        Parameters:
        ----------

        basis_names : list of str
            Names of the basis kets in the same order as the rows of eigvecs.
        eigvecs : np.ndarray, shape (N, N)
            Columns are the eigenvectors |ψ_i> expressed in the original basis.
        energies : array-like, optional
            If provided, labels the eigenstates by energy order.
        tol : float
            Threshold below which coefficients are treated as zero.

        """
        N = eigvecs.shape[0]

        assert eigvecs.shape == (N, N), "eigvecs must be a square matrix"
        #assert len(basis_labels) == N, "basis_labels length must match eigvecs size"

        # if energies are given, label states by E; otherwise by index
        labels = []
        if energies is not None:
            for i, E in enumerate(energies):
                labels.append(f"|ψ_{i}>, E={E:.3f}")

        else:
            labels = [f"|ψ_{i}>" for i in range(1,N+1)]

        for j, basis_label in enumerate(basis_labels):
            coeffs = eigvecs[j, :]
            terms = []
            for i, c in enumerate(coeffs):
                if abs(c) < tol:
                    continue
                # format complex; drop imaginary part if ~= 0
                if abs(c.imag) < tol:
                    c_str = f"{c.real:.3f}"
                else:
                    c_str = f"({c.real:.3f} + {c.imag:.3f}j)"
                terms.append(f"{c_str} {labels[i]}")
                expansion = " + ".join(terms) if terms else "0"
                print(f"{basis_label} = {expansion}")            

        
    # ------------------------------
    # Operator embedding
    # ------------------------------
    def embed_operator(self, op, position):
        """
        Embed a local operator into the full Hilbert space.
        position = index in self.sub_dims (0=cavity, 1=q1, 2=v1, 3=q2, 4=v2,...)
        """
        ops = []
        for i, d in enumerate(self.sub_dims):
            if i == position:
                ops.append(op)
            else:
                ops.append(np.eye(d))
        return self._tensor(*ops)

    def _tensor(self, *ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    # ------------------------------
    # Local operators
    # ------------------------------
    def creation(self, dim):
        op = np.zeros((dim, dim))
        for n in range(dim-1):
            op[n+1, n] = np.sqrt(n+1)
        return op

    def annihilation(self, dim):
        return self.creation(dim).T

    def sigma_plus(self):
        return np.array([[0, 0], [1, 0]])

    def sigma_minus(self):
        return np.array([[0, 1], [0, 0]])

    def sigma_z(self):
        return np.array([[1, 0], [0, -1]])
    
    # ------------------------------
    # Time-domain evolution
    # ------------------------------
    def compute_expectation_value(self, operator, state):
        """
        Compute the expectation value of an operator in a given state.
        operator: numpy array representing the operator
        state: numpy array representing the state vector
        """
        return np.real(np.vdot(state, operator @ state)[0, 0])
    
    def commutator(self, op1, op2):
        """
        Compute the commutator of two operators.
        op1, op2: numpy arrays representing the operators
        """
        return op1 @ op2 - op2 @ op1
    
    def liouville_rhs(self, density_matrix, hamiltonian, hbar=1.0):
        """
        Compute the right-hand side of the Liouville equation.
        state: numpy array representing the density matrix
        hamiltonian: numpy array representing the Hamiltonian operator
        """
        return -1j / hbar * self.commutator(hamiltonian, density_matrix) 
    
    def rk4_step(self, hamiltonian, density_matrix, dt, hbar=1.0):
        """
        Perform a single step of the Runge-Kutta 4th order method for time evolution.
        density_matrix: numpy array representing the current density matrix
        hamiltonian: numpy array representing the Hamiltonian operator
        dt: time step
        """
        #print(f"Density matrix diagonals before RK4 step: {np.diag(density_matrix)}")
        k1 = self.liouville_rhs(density_matrix, hamiltonian, hbar)
        k2 = self.liouville_rhs(density_matrix + 0.5 * dt * k1, hamiltonian, hbar)
        k3 = self.liouville_rhs(density_matrix + 0.5 * dt * k2, hamiltonian, hbar)
        k4 = self.liouville_rhs(density_matrix + dt * k3, hamiltonian, hbar)
        
        new_dm = density_matrix + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        #print(f"Density matrix diagonals after RK4 step: {np.diag(new_dm)}")
        return new_dm
        

