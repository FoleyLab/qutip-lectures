from htc_numpy import HTC_Numpy

params = {
'cav_dim' : 2,
'vib_dim' : 2,
'n_qubits' : 2,
'n_vib' : 2, # 1 vibrational mode per qubit
'g' : 0.1,
'omega_cav' : 1.0,
'omega_vib' : 0.1,
'omega_qubit' : 1.0,
'lambda_vib' : 0.1,
}

# Example: 1 cavity, 2 qubits, 2 vib modes, truncation=2
model = HTC_Numpy(params)

print("Hilbert space dimension:", model.dim)   # 32
print("First 5 labels:", model.labels[:5])

# Get a basis vector
psi = model.basis_state(0)
print("Basis[0] =", model.labels[0], psi.T)

# Embed cavity annihilation operator
a = model.annihilation(model.cav_dim)
a_full = model.embed_operator(a, position=0)   # position=0 is cavity
print("a_full shape:", a_full.shape)

# Embed qubit 1 raising operator
sigma_plus = model.sigma_plus()
sigma_plus_q1 = model.embed_operator(sigma_plus, position=1)

