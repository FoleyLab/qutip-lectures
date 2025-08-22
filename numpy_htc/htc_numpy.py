import numpy as np
from itertools import product

class HTC_Numpy:
    def __init__(self, parameters): 
        """"
            parameters: dictionary containing the following
            cav_dim : truncation dimension of cavity bosonic mode
            vib_dim : truncation dimension of vibrational bosonic modes
            n_qubits : number of qubits
            n_vib : number of vibrational modes
            g : coupling between cavity and qubits
            omega_cav : frequency of cavity mode
            omega_vib : frequency of vibrational modes
            omega_qubit : frequency of qubits
            lambda_vib : coupling strength of vibrational modes

        """
        # Unpack parameters
        self.cav_dim = parameters.get('cav_dim', 2)
        self.vib_dim = parameters.get('vib_dim', 2)
        self.n_qubits = parameters.get('n_qubits', 2)
        self.n_vib = parameters.get('n_vib', 2)


        self.g = parameters.get('g', 1.0)
        self.omega_cav = parameters.get('omega_cav', 1.0)
        self.omega_vib = parameters.get('omega_vib', 0.1)
        self.omega_qubit = parameters.get('omega_qubit', 1.0)
        self.lambda_vib = parameters.get('lambda_vib', 0.1)

        self.qubit_dim = 2  # Assuming qubits are two-level systems

        # ordering: [cavity, q1, v1, q2, v2, ...]
        self.sub_dims = [self.cav_dim] + [self.qubit_dim, self.vib_dim] * self.n_qubits
        self.dim = np.prod(self.sub_dims)
        print(f"Total Hilbert space dimension: {self.dim}")
        print(f"coupling strength g: {self.g}")
        # Build basis labels
        self.labels = self._build_labels()

    # ------------------------------
    # Basis and labels
    # ------------------------------
    def _build_labels(self):
        cav_labels   = [f"{n}" for n in range(self.cav_dim)]
        qubit_labels = ["g", "e"] if self.qubit_dim == 2 else [str(i) for i in range(self.qubit_dim)]
        vib_labels   = [f"{n}" for n in range(self.vib_dim)]

        labels = []
        # loop explicitly in the correct order
        for cav in cav_labels:
            for q1 in qubit_labels:
                for v1 in vib_labels:
                    for q2 in qubit_labels:
                        for v2 in vib_labels:
                            label = f"|{cav}, {q1}, {v1}, {q2}, {v2}>"
                            labels.append(label)
        return labels

    def basis_state(self, index):
        """Return computational basis vector for given index"""
        state = np.zeros((self.dim, 1))
        state[index, 0] = 1
        return state

    def index_of(self, label):
        """Return basis index for given label string"""
        return self.labels.index(label)

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
        print(f"Density matrix diagonals before RK4 step: {np.diag(density_matrix)}")
        k1 = self.liouville_rhs(density_matrix, hamiltonian, hbar)
        k2 = self.liouville_rhs(density_matrix + 0.5 * dt * k1, hamiltonian, hbar)
        k3 = self.liouville_rhs(density_matrix + 0.5 * dt * k2, hamiltonian, hbar)
        k4 = self.liouville_rhs(density_matrix + dt * k3, hamiltonian, hbar)
        
        new_dm = density_matrix + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        print(f"Density matrix diagonals after RK4 step: {np.diag(new_dm)}")
        return new_dm
        

