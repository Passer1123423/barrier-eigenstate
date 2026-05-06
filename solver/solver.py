import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def solver_run(Vx0 = [-3, -2, 2, 3], V_val = [0, 0], n_show = [0,1,2,3],num_states = 50):
    hbar = 1.05457E-34 #J/s
    m = 9.1093E-31 #kg

    x_min = -8
    x_max = 8
    N = 2000

    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]

    V = np.zeros(N)

    mask1 = (x >= Vx0[0]) & (x < Vx0[1])
    mask2 = (x > Vx0[2]) & (x <= Vx0[3])

    V[mask1] = V_val[0]
    V[mask2] = V_val[1]

    # Hamiton operator
    coef = hbar * hbar / (2 * m) / dx**2  * 6.242E+18 * 1E+18
    main_diag = 2 * coef + V
    off_diag = -coef * np.ones(N-1)
    H = diags(
        diagonals=[off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        format='csr'
    )
    vals, vecs = eigsh(H, k=num_states,which="SA")

    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    # 取前4个能级和波函数

    E = vals[:num_states]
    psi = vecs[:, :num_states].astype(complex)
    psit = psi

    # 归一化波函数：sum |psi|^2 dx = 1
    for n in range(num_states):
        norm = np.sqrt(np.sum(np.abs(psi[:, n]) ** 2) * dx)
        psi[:, n] = psi[:, n] / norm
        if psi[N // 2, n] < 0:
            psi[:, n] *= -1

    return x,E,psi,V

    # 画势垒
    """plt.figure(figsize=(10, 6))
    for n in range(4):
        if n in n_show:
            plt.plot(
                x,
                np.real(psit[:, n]),
                label=f"n={n}, E={E[n]:.4f}eV"
            )
            plt.axhline(E[n], linestyle="--", linewidth=0.8)
    plt.plot(x, V, color="black", linewidth=2, label="V(x)")

    plt.xlabel("x")
    plt.ylabel("psi_real")
    plt.title("First 4 eigenstates")
    plt.legend()
    plt.grid(True)
    plt.show()"""

if __name__ == "__main__":
    V_val = [1,1]
    x,E,psi,V = solver_run(V_val=V_val)
    print(E)

