import numpy as np
import sys
sys.path.append("/home/jfoley19/Code/qed-ci/src/")
#sys.path.append("/home/nvu12/software/qed_ci_main/qed_ci_112123/qed-ci/src/")
#np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
from helper_PFCI import PFHamiltonianGenerator
from helper_PFCI import Determinant
from helper_cqed_rhf import cqed_rhf
from nuclear_grad import *
np.set_printoptions(threshold=sys.maxsize)

# conversion factor
BOHR_TO_ANGSTROM = 0.52917721092

options_dict = {'basis': '6-31g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10
                  }


cavity_options = {
    'omega_value' : 0.12086,
    'lambda_vector' : np.array([0.0, 0.0, 0.05]),
    'ci_level' : 'cas',
    'number_of_photons' : 1,
    'photon_number_basis' : False,
    'canonical_mos' : False,
    'coherent_state_basis' : True,
    'spin_adaptation': "singlet",
    'davidson_roots' : 3,
    'davidson_threshold' : 1e-7,
    'davidson_maxdim': 8,
    'davidson_maxiter':100,
    'davidson_indim':6,
    'nact_orbs' : 4,
    'nact_els' : 4
}
psi4.set_options(options_dict)

# geometry template
mol_tmpl = """
0 1
Li
H 1 **R**
symmetry c1
no_reorient
nocom
"""

# central geometry in Angstroms 
r_center = 1.0

# displacement in angstroms
h_ang = 0.002

# displacement in Bohr 
h_bohr = h_ang / BOHR_TO_ANGSTROM

# array of bond lengths in Angstrom to go into the geometries
r_vals = np.array([r_center - 2 * h_ang, r_center - h_ang, r_center, r_center + h_ang, r_center + 2 * h_ang])

# coefficients to multply the energies at each bond length by
np.array([-1, 8, -8, 1]) / (12 * h_bohr)

E_array = np.zeros((cavity_options['davidson_roots'],4))

# loop through bond lengths and compute energies
for idx, r in enumerate(r_vals):
    mol_str = mol_tmpl.replace("**R**", f"{r:.6f}")
    H2_PF = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
    for i in range(nstates):
        E_array[i, idx] = H2_PF.CASSCFeigs[i]
    print(f"Computed energies at R = {r:.6f} Å")

# compute finite difference derivatives (gradients)
gradients = np.dot(E_array, coeffs)

# print results
for i in range(nstates):
    print(f"State {i} dE/dR = {gradients[i]:.10f} Hartree/Bohr")


                    