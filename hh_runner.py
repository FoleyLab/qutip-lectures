from htc_class import MultiPartiteSystem

# Example parameters
system = MultiPartiteSystem(
    w_q1=5.0, w_q2=5.5,
    w_ho1=1.0, w_ho2=1.2, w_hoc=0.9,
    g_q1_ho1=0.05, g_q2_ho2=0.06,
    g_q1_hoc=0.02, g_q2_hoc=0.02,
    N_ho=8  # truncation
)

H = system.get_hamiltonian()
print(H)

