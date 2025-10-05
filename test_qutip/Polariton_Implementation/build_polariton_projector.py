import numpy as np
from qutip import *
from qutip import qeye, destroy, tensor
from qutip import Qobj

# Identities
Iq = qeye(2)
Ic = qeye(2) 


# Bosonic operators
am = destroy(2)
ap = am.dag()
nc = ap * am

# Pauli operators
sz = sigmaz()
sm = destroy(2)
sp = sm.dag()

# omega values
wc = 0.120865  # cavity frequency
wq = wc  # qubit frequency
g = 0.01  # coupling strength

# build JC Hamiltonian
H = wc * tensor(Iq, nc) - 0.5 * wq * tensor(sz, Ic) + g * (tensor(sp, am) + tensor(sm, ap))

H_np = H.full()
print(H_np)
print(type(H_np))
print(np.shape(H_np))

vals, vecs = np.linalg.eigh(H_np)


H_bar = np.zeros((4, 4), dtype=complex)
# loop over eigenvectors and associated eigenvalues
for i in range(vecs.shape[1]):
    print(f"Eigenvalue {i}: {vals[i]}")
    v = vecs[:, i][:, np.newaxis]  # Extract the i-th vector as a column vector
    print(v @ v.T.conj())  # Print the outer product of v and its conjugate transpose
    H_bar += vals[i] * (v @ v.T.conj())

print(H_bar)