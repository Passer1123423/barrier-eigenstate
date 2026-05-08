import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from solver.solver import solver_run

def gaussian_wavepacket(x, x0=5.0, sigma=0.8, k0=-5.0):
    """
    一维高斯波包

    x     : 已有的空间坐标数组
    x0    : 波包中心位置
    sigma : 空间宽度，也就是标准差量级
    k0    : 平均波数，对应平均动量 p0 = hbar * k0
    """
    psi = np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * x)

    # 按离散空间归一化，使 sum(|psi|^2) dx = 1
    dx = x[1] - x[0]
    psi = psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)

    return psi

def Animat(x,E,psi,V,n_show=[0,1]):
    Et = list()
    for n in n_show:
        Et.append(E[n])
    Et = np.array(Et, dtype=float)
    Et = ((Et ** -1))
    T = np.max(Et)

    dx = x[1] - x[0]

    theta_list = np.linspace(0, 2 * np.pi * T, 120)
    fig, ax = plt.subplots(figsize=(10, 6))
    axV = ax.twinx()

    line_re = dict.fromkeys(n_show,None)
    line_im = dict.fromkeys(n_show,None)
    line_prob = dict.fromkeys(n_show,None)
    for n in n_show:
        psi[:,n] /= np.sqrt(np.sum(np.abs(psi[:,n]) ** 2)*dx)
        line_re[n], = ax.plot(x, np.real(psi[:,n]), label=f"Re ψ{n}")
        line_im[n], = ax.plot(x, np.imag(psi[:,n]), label=f"Im ψ{n}")
        line_prob[n], = ax.plot(x, np.abs(psi[:,n]) ** 2, label=f"|ψ|²{n}")

    ax.set_xlabel("x")
    ax.set_ylabel("ψ")
    ax.legend()
    ax.grid(True)
    axV.plot(x, V, color='black')
    axV.set_ylabel("Energy / eV")

    ymax = max(
        np.max(np.abs(np.real(psi[:,n_show]))),
        np.max(np.abs(np.imag(psi[:,n_show]))),
        np.max(np.abs(psi[:,n_show]) ** 2)
    )
    ax.set_ylim(-1.2 * ymax, 1.2 * ymax)
    axV.set_ylim(-3, 3)

    def update(frame):
        theta = theta_list[frame]
        psi_t = psi.copy()

        for n in n_show:
            psi_t[:, n] = psi[:, n] * np.exp(-1j * E[n] * theta)
            line_re[n].set_ydata(np.real(psi_t[:,n]))
            line_im[n].set_ydata(np.imag(psi_t[:,n]))
            line_prob[n].set_ydata(np.abs(psi_t[:,n]) ** 2)

        ax.set_title(f"Stationary state, time = {theta * 6.582E-1:.2f}fs ")

        return line_re, line_im, line_prob

    ani = FuncAnimation(fig, update, frames=len(theta_list), interval=50, blit=False)

    plt.show()
    ani.save("figures/wave.gif", writer="pillow", fps=20)

def Animat_super(x,E,psi,V,n_show=[0,1],c=[1,1]):
    dx = x[1] - x[0]

    theta_list = np.linspace(0, 2 * np.pi, 120)
    fig, ax = plt.subplots(figsize=(10, 6))
    axV = ax.twinx()
    surpsi = c[0]*psi[:,n_show[0]] + c[1]*psi[:,n_show[1]]
    surpsi = surpsi / np.sqrt(np.sum(np.abs(surpsi) ** 2)*dx)
    prob = np.abs(surpsi) ** 2
    line_prob, = ax.plot(x, np.abs(surpsi) ** 2, label="|ψ_super|²")

    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")
    ax.legend()
    ax.grid(True)
    axV.plot(x, V, color='black')
    axV.set_ylabel("Energy / eV")

    ymax = max(prob)
    ax.set_ylim(-1.2 * ymax, 1.2 * ymax)
    axV.set_ylim(-3, 3)

    def update(frame):
        theta = theta_list[frame]
        psi_t = psi.copy()

        psi_t[:, 1] = psi[:, 1] * np.exp(-1j * theta)

        surpsi_t = c[0]*psi_t[:,n_show[0]] + c[1]*psi_t[:,n_show[1]]
        surpsi_t = surpsi_t / np.sqrt(np.sum(np.abs(surpsi_t) ** 2)*dx)
        line_prob.set_ydata(np.abs(surpsi_t) ** 2)

        ax.set_title(f"Stationary state, phase = {theta} ")

        return line_prob

    ani = FuncAnimation(fig, update, frames=len(theta_list), interval=50, blit=False)
    plt.show()
    ani.save("figures/surwave.gif", writer="pillow", fps=20)

def Animat_kspace(x,E,psi,n_show=[0,1]):
    Et = list()
    for n in n_show:
        Et.append(E[n])
    Et = np.array(Et, dtype=float)
    Et = ((Et ** -1))
    T = np.max(Et[0])

    N = len(x)
    dx = x[1] - x[0]
    k = 2*np.pi*np.fft.fftfreq(N, d=dx)
    k = np.fft.fftshift(k)
    dk = k[1] - k[0]

    psi_k = psi.copy()
    for n in n_show:
        psi_k[:,n] = np.einsum('ij,j->i',np.exp(-1j * k[:,None] * x[None,:]), psi[:,n]) * dx
        psi_k[:,n] = psi_k[:,n]/np.sqrt(np.sum(np.abs(psi_k[:,n])**2)*dk)


    theta_list = np.linspace(0, 2 * np.pi * T, 120)
    fig, ax = plt.subplots(figsize=(10, 6))

    line_re = dict.fromkeys(n_show, None)
    line_im = dict.fromkeys(n_show, None)
    line_prob = dict.fromkeys(n_show, None)

    for n in n_show:
        line_re[n], = ax.plot(k, np.real(psi_k[:,n]), label=f"Re ψ_k{n}")
        line_im[n], = ax.plot(k,np.imag(psi_k[:,n]),label=f"Im ψ_k{n}")
        line_prob[n], = ax.plot(k,np.abs(psi_k[:,n])**2,label=f"P ψ_k{n}")

    ax.set_xlabel("k")
    ax.set_ylabel("ψ_k")
    ax.legend()
    ax.grid(True)

    ymax = np.max(np.abs(np.abs(psi_k[:,n_show])**2))
    ax.set_xlim(-20, 20)
    ax.set_ylim(-1.2 * ymax, 1.2 * ymax)

    def update(frame):
        theta = theta_list[frame]
        psi_t = psi.copy()

        for n in n_show:
            psi_t[:, n] = psi[:, n] * np.exp(-1j * E[n] * theta)
            psi_k[:,n] = np.einsum('ij,j->i',np.exp(-1j * k[:,None] * x[None,:]), psi_t[:,n]) * dx
            psi_k[:,n] = psi_k[:,n]/np.sqrt(np.sum(np.abs(psi_k[:,n])**2)*dk)
            line_re[n].set_ydata(np.real(psi_k[:,n]))
            line_im[n].set_ydata(np.imag(psi_k[:,n]))
            line_prob[n].set_ydata(np.abs(psi_k[:,n])**2)

        ax.set_title(f"Stationary state, phase = {theta * 6.582E-1:.2f}fs ")

        return list(line_re.values()), list(line_im.values()), list(line_prob.values())

    ani = FuncAnimation(fig, update, frames=len(theta_list), interval=50, blit=False)

    plt.show()
    ani.save("figures/wave_kspace.gif", writer="pillow", fps=20)

def Scatter(x,E,psi,psi_in,cut_state=200):
    Et = E.copy()
    Et = ((Et ** -1))
    T = 10

    dx = x[1] - x[0]

    cn = list([0 for n in range(cut_state)])
    for n in range(cut_state):
        cn[n] = np.vdot(psi[:,n], psi_in) * dx

    theta_list = np.linspace(0, 2 * np.pi * T, 600)
    fig, ax = plt.subplots(figsize=(10, 6))
    figv, bx = plt.subplots(figsize=(10, 6))
    axV = ax.twinx()

    line_prob, = ax.plot(x, np.abs(psi_in) ** 2, label="|ψ|²")
    xm = list()
    xm.append(np.inner(x,np.abs(psi_in)**2)*dx)
    ax.set_xlabel("x")
    ax.set_ylabel("ψ")
    ax.legend()
    ax.grid(True)
    axV.plot(x, V, color='black')
    axV.set_ylabel("Energy / eV")
    ymax = np.max(np.abs(psi_in) ** 2)
    ax.set_ylim(-1.2 * ymax, 1.2 * ymax)
    axV.set_ylim(-3,3)

    def update(frame):
        theta = theta_list[frame]
        psi_t = psi.copy()
        psi_in_t = np.zeros_like(psi_in,dtype=complex)
        for n in range(cut_state):
            psi_t[:,n] = psi[:, n] * np.exp(-1j * E[n] * theta)
            psi_in_t += cn[n] * psi_t[:,n]
        psi_in_t /= np.sqrt(np.sum(np.abs(psi_in_t)**2)*dx)
        xm.append(np.inner(x, np.abs(psi_in_t) ** 2) * dx)
        line_prob.set_ydata(np.abs(psi_in_t)**2)

        ax.set_title(f"Stationary state, time = {theta * 6.582E-1:.2f}ps ")

        return line_prob

    ani = FuncAnimation(fig, update, frames=len(theta_list), interval=50, blit=False)

    plt.show()
    ani.save("figures/scatter.gif", writer="pillow", fps=20)

    xm = np.array(xm)
    idt = np.arange(xm.size).reshape(xm.shape)
    bx.set_xlabel("t")
    bx.set_ylabel("xm")
    bx.legend()
    bx.grid(True)
    bx.plot(idt, xm)
    figv.savefig("figures/xm.png")

if __name__ == "__main__":
    V_val = [1, 1]
    x, E, psi, V = solver_run(num_states=500)
    print(E)
    #Animat(x, E, psi, n_show=[1], V=V)
    #Animat_super(x,E,psi,V=V)
    #Animat_kspace(x,E,psi,n_show=[0])
    psi_in = gaussian_wavepacket(x)
    Scatter(x,E,psi,psi_in=psi_in,cut_state=500)
