import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, concurrence, destroy, expect, fidelity,
                   ket2dm, mesolve, ptrace, qeye, sigmaz,
                   tensor)
#from qutip_qip.operations import phasegate, sqrtiswap
from scipy.special import sici

N = 10

wc = 5.0 * 2 * np.pi
w1 = 3.0 * 2 * np.pi
w2 = 2.0 * 2 * np.pi

g1 = 0.01 * 2 * np.pi
g2 = 0.0125 * 2 * np.pi

tlist = np.linspace(0, 100, 500)

width = 0.5

# resonant SQRT iSWAP gate
T0_1 = 20
T_gate_1 = (1 * np.pi) / (4 * g1)

# resonant iSWAP gate
T0_2 = 60
T_gate_2 = (2 * np.pi) / (4 * g2)

# cavity operators
a = tensor(destroy(N), qeye(2), qeye(2))
n = a.dag() * a

# operators for qubit 1
sm1 = tensor(qeye(N), destroy(2), qeye(2))
sz1 = tensor(qeye(N), sigmaz(), qeye(2))
n1 = sm1.dag() * sm1

# oeprators for qubit 2
sm2 = tensor(qeye(N), qeye(2), destroy(2))
sz2 = tensor(qeye(N), qeye(2), sigmaz())
n2 = sm2.dag() * sm2

# Hamiltonian using QuTiP
Hc = a.dag() * a
H1 = -0.5 * sz1
H2 = -0.5 * sz2
Hc1 = g1 * (a.dag() * sm1 + a * sm1.dag())
Hc2 = g2 * (a.dag() * sm2 + a * sm2.dag())

H = wc * Hc + w1 * H1 + w2 * H2 + Hc1 + Hc2

print(H)

# initial state: start with one of the qubits in its excited state
psi0 = tensor(basis(N, 0), basis(2, 1), basis(2, 0))

def step_t(w1, w2, t0, width, t):
    """
    Step function that goes from w1 to w2 at time t0
    as a function of t.
    """
    return w1 + (w2 - w1) * (t > t0)


fig, axes = plt.subplots(1, 1, figsize=(8, 2))
axes.plot(tlist, [step_t(0.5, 1.5, 50, 0.0, t) for t in tlist], "k")
axes.set_ylim(0, 2)
fig.tight_layout()

def wc_t(t, args=None):
    return wc


def w1_t(t, args=None):
    return (
        w1
        + step_t(0.0, wc - w1, T0_1, width, t)
        - step_t(0.0, wc - w1, T0_1 + T_gate_1, width, t)
    )


def w2_t(t, args=None):
    return (
        w2
        + step_t(0.0, wc - w2, T0_2, width, t)
        - step_t(0.0, wc - w2, T0_2 + T_gate_2, width, t)
    )


H_t = [[Hc, wc_t], [H1, w1_t], [H2, w2_t], Hc1 + Hc2]

res = mesolve(H_t, psi0, tlist, [], [])

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

axes[0].plot(
    tlist,
    np.array(list(map(wc_t, tlist))) / (2 * np.pi),
    "r",
    linewidth=2,
    label="cavity",
)
axes[0].plot(
    tlist,
    np.array(list(map(w1_t, tlist))) / (2 * np.pi),
    "b",
    linewidth=2,
    label="qubit 1",
)
axes[0].plot(
    tlist,
    np.array(list(map(w2_t, tlist))) / (2 * np.pi),
    "g",
    linewidth=2,
    label="qubit 2",
)
axes[0].set_ylim(1, 6)
axes[0].set_ylabel("Energy (GHz)", fontsize=16)
axes[0].legend()

axes[1].plot(tlist, np.real(expect(n, res.states)), "r",
             linewidth=2, label="cavity")
axes[1].plot(tlist, np.real(expect(n1, res.states)), "b",
             linewidth=2, label="qubit 1")
axes[1].plot(tlist, np.real(expect(n2, res.states)), "g",
             linewidth=2, label="qubit 2")
axes[1].set_ylim(0, 1)

axes[1].set_xlabel("Time (ns)", fontsize=16)
axes[1].set_ylabel("Occupation probability", fontsize=16)
axes[1].legend()

fig.tight_layout()