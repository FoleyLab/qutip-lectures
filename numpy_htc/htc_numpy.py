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
                Truncation dimension of cavity bosonic mode
            dim_vib : int
                Truncation dimension of vibrational bosonic modes
            n_qubits : int
                Number of qubits
            n_vib_per_qubit : int
                Number of vibrational modes per qubit

            # Uncoupled basis parameters
            coupling_cavity_qubit : float
                Cavity–qubit coupling strength (g)
            freq_cavity : float
                Cavity mode frequency (ω_cav)
            freq_qubit : float
                Qubit frequency (ω_q)
            freq_vib_qubit : float
                Vibrational mode frequency for qubit vibrations (ω_v)
            coupling_qubit_vib : float
                Qubit–vibration coupling (Huang-Rhys factor, λ)

            # Polaritonic basis parameters
            polariton_energies : list of float
                Energies of polariton states (default: [0, 0.9, 1.1, 2.0])
            polariton_vib_frequencies : list of float
                Vibrational frequencies on each polariton surface
            polariton_vib_couplings : list of float
                Vibronic couplings on each polariton surface
            coupling_vib_polaritonic : float
                Global vibronic coupling in the polaritonic basis
        """

        # --- Hilbert space dimensions ---
        self.dim_qubit = 2  # Assuming qubits are two-level systems
        self.dim_cavity = params.get('dim_cavity', 2)
        self.dim_vib = params.get('dim_vib', 2)
        self.n_qubits = params.get('n_qubits', 2)
        self.n_vib_per_qubit = params.get('n_vib_per_qubit', 1)
        self.n_vib_per_cavity = params.get('n_vib_per_cavity', 1)

        # --- Uncoupled basis parameters (incomplete) ---
        self.freq_cavity = params.get('freq_cavity', 1.0)
        self.cavity_lambda = params.get('cavity_lambda', 0.1)  # Cavity displacement
        self.freq_qubit = params.get('freq_qubit', 1.0)
        self.freq_vib_qubit = params.get('freq_vib_qubit', 0.1)
        self.freq_vib_cavity = params.get('freq_vib_cavity', 0.1)
        self.qubit_dipole_values = params.get('qubit_dipole_values', [1.0, 1.0, 1.0]) # [μ_gg, μ_eg, μ_ee]
        self.qubit_dipole_matrix = np.array([[self.qubit_dipole_values[0], self.qubit_dipole_values[1]],
                                             [self.qubit_dipole_values[1], self.qubit_dipole_values[2]]])
        
        self.qubit_d_matrix = self.cavity_lambda * self.qubit_dipole_matrix
        

        #

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





        # ordering: [q1, c, q2, v1, vc, v2] 
        self.sub_dims = [self.dim_qubit, self.dim_cavity, self.dim_qubit, self.dim_vib, self.dim_vib, self.dim_vib]
        self.dim = np.prod(self.sub_dims)
        print(f"Total Hilbert space dimension: {self.dim}")
        print(f"cavity coupling λ: {self.cavity_lambda}")
        print(f"qubit dipole matrix:\n{self.qubit_d_matrix}")
        print(f"Qubit frequency is {self.freq_qubit}, cavity frequency is {self.freq_cavity}, vib frequency is {self.freq_vib_qubit}")
        # Build basis labels
        self.labels = self._build_labels()

    # ------------------------------
    # Basis and labels for uncoupled basis
    # ------------------------------
    def _build_labels(self):
        cav_labels   = [f"{n}" for n in range(self.dim)]
        qubit_labels = ["g", "e"] if self.dim_qubit == 2 else [str(i) for i in range(self.dim_qubit)]
        vib_labels   = [f"{n}" for n in range(self.dim_vib)]

        labels = []
        if self.n_qubits == 2 and self.n_vib_per_qubit * self.n_qubits == 2:
            # loop explicitly in the correct order
            for cav in cav_labels:
                for q1 in qubit_labels:
                    for v1 in vib_labels:
                        for q2 in qubit_labels:
                            for v2 in vib_labels:
                                label = f"|{cav}, {q1}, {v1}, {q2}, {v2}>"
                                labels.append(label)
        elif self.n_qubits == 1 and self.n_vib_per_qubit * self.n_qubits == 1:
            for cav in cav_labels:
                for q1 in qubit_labels:
                    for v1 in vib_labels:
                        label = f"|{cav}, {q1}, {v1}>"
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

        return H  
    
    def build_polariton_vibrational_hamiltonian(self):
        """
        Build the Hamiltonian according to
        H_pol-vib = |pol1><pol1| ω_v,1 b†b
                  + |pol2><pol2| ω_v,2 b†b
                  + |pol3><pol3| ω_v,3 b†b
                  + |pol4><pol4| ω_v,4 b†b
        where b†b are the creation and annihilation operators for the vibrational modes on each qubit.
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

        H  = np.kron(pol1_proj, self.freq_vib_pol1 * n_vib)
        H += np.kron(pol2_proj, self.freq_vib_pol2 * n_vib)
        H += np.kron(pol3_proj, self.freq_vib_pol3 * n_vib)
        H += np.kron(pol4_proj, self.freq_vib_pol4 * n_vib)

        return H
    
    def build_polariton_vib_coupling(self):
        """
        Build the vibronic coupling Hamiltonian in the polaritonic basis.
        H_coupling = λ_12 |pol1><pol2| (b† + b)
                   + λ_13 |pol1><pol3| (b† + b)
                   + λ_14 |pol1><pol4| (b† + b)
                   + h.c.
        where λ_ij are the vibronic couplings between polaritonic states.
        """
        # build (b† + b) operator for vibrational mode
        b_dagger = self.creation(self.dim_vib)
        b = self.annihilation(self.dim_vib)
        vib_coupling_op = b_dagger + b

        pol_dim = len(self.polariton_energies)
        H = self.coupling_vib_pol12 * (
            np.kron(self.build_projector(pol_dim, 0, 1), vib_coupling_op) + 
            np.kron(self.build_projector(pol_dim, 1, 0), vib_coupling_op) 
        )
        H += self.coupling_vib_pol13 * (
            np.kron(self.build_projector(pol_dim, 0, 2), vib_coupling_op) +
            np.kron(self.build_projector(pol_dim, 2, 0), vib_coupling_op)
        )
        H += self.coupling_vib_pol14 * (
            np.kron(self.build_projector(pol_dim, 0, 3), vib_coupling_op) + 
            np.kron(self.build_projector(pol_dim, 3, 0), vib_coupling_op) 
        )

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
        

