import numpy as np
import qutip as qt
from scipy.linalg import eigh

# System parameters
n_vib = 5  # vibrational states per molecule
n_cav = 5  # cavity states
n_mol = 2  # number of molecules

# Physical parameters
omega_v = [1.0, 1.0]  # vibrational frequencies for each molecule
omega_e = [2.0, 2.0]  # electronic transition frequencies
lambda_param = [0.5, 0.5]  # Huang-Rhys factors
omega_c = 2.0  # cavity frequency
g_cav = 0.1  # cavity coupling strength

def build_single_molecule_holstein(omega_v, omega_e, lambda_param, n_vib):
    """Build Holstein Hamiltonian for a single molecule"""
    # Electronic operators (2-level system)
    sigma_z = qt.sigmaz()
    sigma_plus = qt.sigmap()
    sigma_minus = qt.sigmam()
    
    # Vibrational operators
    a_vib = qt.destroy(n_vib)
    a_vib_dag = qt.create(n_vib)
    n_vib_op = qt.num(n_vib)
    x_vib = (a_vib + a_vib_dag)  # position operator (dimensionless)
    
    # Identity operators
    id_elec = qt.qeye(2)
    id_vib = qt.qeye(n_vib)
    
    # Build Holstein Hamiltonian in tensor product space
    # H = ℏω_v b†b + ℏω_e σ†σ - ℏλω_v(b† + b)σ†σ
    
    # Vibrational term: ℏω_v b†b ⊗ I_elec
    H_vib = qt.tensor(id_elec, omega_v * n_vib_op)
    
    # Electronic term: ℏω_e σ†σ ⊗ I_vib  
    # Note: σ†σ = (I + σ_z)/2 for excited state population
    sigma_dag_sigma = (qt.qeye(2) + sigma_z) / 2
    H_elec = qt.tensor(sigma_dag_sigma, id_vib) * omega_e
    
    # Vibronic coupling: -ℏλω_v(b† + b) σ†σ
    H_vibronic = -qt.tensor(sigma_dag_sigma, lambda_param * omega_v * x_vib)
    
    H_holstein = H_vib + H_elec + H_vibronic
    
    return H_holstein

def get_polaron_basis(H_holstein):
    """Diagonalize Holstein Hamiltonian to get polaron basis"""
    # Get eigenvalues and eigenvectors
    eigenvals, eigenvecs = H_holstein.eigenstates()
    
    return eigenvals, eigenvecs

def transform_operator_to_polaron_basis(operator, eigenvecs):
    """Transform an operator to the polaron basis"""
    # U† O U where U is the matrix of eigenvectors
    U = qt.Qobj(eigenvecs)
    return U.dag() * operator * U

# Build single-molecule Holstein Hamiltonians
print("Building single-molecule Holstein Hamiltonians...")
H_mol = []
polaron_energies = []
polaron_states = []

for i in range(n_mol):
    H_i = build_single_molecule_holstein(omega_v[i], omega_e[i], lambda_param[i], n_vib)
    energies, states = get_polaron_basis(H_i)
    
    H_mol.append(H_i)
    polaron_energies.append(energies)
    polaron_states.append(states)
    
    print(f"Molecule {i+1}: Ground state energy = {energies[0]:.4f}")
    print(f"Molecule {i+1}: First excited polaron energy = {energies[1]:.4f}")

# Build operators in full Hilbert space (mol1 ⊗ mol2 ⊗ cavity)
print("\nBuilding full system operators...")

# Identity operators for each subsystem
id_mol1 = qt.qeye(2 * n_vib)  # 2-level × n_vib states
id_mol2 = qt.qeye(2 * n_vib)
id_cav = qt.qeye(n_cav)

# Holstein Hamiltonians in full space
H_mol1_full = qt.tensor(H_mol[0], id_mol2, id_cav)
H_mol2_full = qt.tensor(id_mol1, H_mol[1], id_cav)

# Cavity Hamiltonian
a_cav = qt.tensor(id_mol1, id_mol2, qt.destroy(n_cav))
H_cav = omega_c * a_cav.dag() * a_cav

# Electronic operators for cavity coupling
# Need σ+ and σ- for each molecule in full space
def build_sigma_operators():
    """Build electronic ladder operators for each molecule in full space"""
    sigma_plus_mol = []
    sigma_minus_mol = []
    
    for i in range(n_mol):
        # Build σ+ for molecule i in the single-molecule space
        sigma_p_single = qt.tensor(qt.sigmap(), qt.qeye(n_vib))
        
        if i == 0:
            # Molecule 1: σ+ ⊗ I_mol2 ⊗ I_cav
            sigma_p_full = qt.tensor(sigma_p_single, id_mol2, id_cav)
        else:
            # Molecule 2: I_mol1 ⊗ σ+ ⊗ I_cav  
            sigma_p_full = qt.tensor(id_mol1, sigma_p_single, id_cav)
            
        sigma_plus_mol.append(sigma_p_full)
        sigma_minus_mol.append(sigma_p_full.dag())
    
    return sigma_plus_mol, sigma_minus_mol

sigma_plus, sigma_minus = build_sigma_operators()

# Light-matter interaction
H_int = g_cav * (sigma_plus[0] * a_cav + sigma_minus[0] * a_cav.dag() + 
                 sigma_plus[1] * a_cav + sigma_minus[1] * a_cav.dag())

# Full Hamiltonian in uncoupled basis
H_full_uncoupled = H_mol1_full + H_mol2_full + H_cav + H_int

print(f"Full Hilbert space dimension: {H_full_uncoupled.shape[0]}")

# Transform to polaron basis
print("\nTransforming to polaron basis...")

# Build transformation matrix for full system
# This is the tensor product of individual polaron transformations
U1 = qt.Qobj(polaron_states[0])  # Molecule 1 polaron states
U2 = qt.Qobj(polaron_states[1])  # Molecule 2 polaron states
U_cav = qt.qeye(n_cav)  # Cavity states unchanged

# Full transformation matrix
U_full = qt.tensor(U1, U2, U_cav)

# Transform Hamiltonian to polaron basis
H_full_polaron = U_full.dag() * H_full_uncoupled * U_full

# Initial state: localized excitation on molecule 1
# Start with |e1, g2, vac_cav⟩ in uncoupled basis
psi_init_uncoupled = qt.tensor(
    qt.tensor(qt.basis(2,1), qt.basis(n_vib,0)),  # |e1, 0_vib⟩
    qt.tensor(qt.basis(2,0), qt.basis(n_vib,0)),  # |g2, 0_vib⟩  
    qt.basis(n_cav,0)  # |0_cav⟩
)

# Transform initial state to polaron basis
psi_init_polaron = U_full.dag() * psi_init_uncoupled

print(f"Initial state overlap with polaron ground state: {abs(psi_init_polaron[0])**2:.4f}")

# Time evolution parameters
t_max = 50.0
n_points = 500
times = np.linspace(0, t_max, n_points)

# Observables in polaron basis
# Electronic population operators
n1_elec = U_full.dag() * qt.tensor(
    qt.tensor((qt.qeye(2) + qt.sigmaz())/2, qt.qeye(n_vib)), 
    id_mol2, id_cav
) * U_full

n2_elec = U_full.dag() * qt.tensor(
    id_mol1,
    qt.tensor((qt.qeye(2) + qt.sigmaz())/2, qt.qeye(n_vib)),
    id_cav
) * U_full

n_cav = U_full.dag() * (a_cav.dag() * a_cav) * U_full

# Run time evolution
print(f"\nRunning time evolution...")
print(f"Time range: 0 to {t_max}")
print(f"Number of time points: {n_points}")

result = qt.mesolve(H_full_polaron, psi_init_polaron, times, 
                   e_ops=[n1_elec, n2_elec, n_cav])

print("Time evolution complete!")
print(f"Final populations - Mol1: {result.expect[0][-1]:.4f}, Mol2: {result.expect[1][-1]:.4f}, Cavity: {result.expect[2][-1]:.4f}")

# The result.expect contains [n1_elec(t), n2_elec(t), n_cav(t)]
# You can now plot these to see the population dynamics!
