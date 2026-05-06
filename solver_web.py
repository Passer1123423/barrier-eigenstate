import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags


def solver_run(Vx0=(-3, -2, 2, 3), V_val=(1, 1), num_states=21, N=800):
    hbar = 1.05457E-34
    m = 9.1093E-31

    x_min = -8
    x_max = 8

    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]

    V = np.zeros(N)

    mask1 = (x >= Vx0[0]) & (x < Vx0[1])
    mask2 = (x > Vx0[2]) & (x <= Vx0[3])

    V[mask1] = V_val[0]
    V[mask2] = V_val[1]

    coef = hbar * hbar / (2 * m) / dx**2 * 6.242E18 * 1E18

    main_diag = 2 * coef + V
    off_diag = -coef * np.ones(N - 1)

    H = diags(
        diagonals=[off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        format="csr"
    )

    vals, vecs = eigsh(H, k=num_states, which="SA")

    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    E = vals[:num_states]
    psi = vecs[:, :num_states].astype(complex)

    for n in range(num_states):
        norm = np.sqrt(np.sum(np.abs(psi[:, n]) ** 2) * dx)
        psi[:, n] /= norm

        if np.real(psi[N // 2, n]) < 0:
            psi[:, n] *= -1

    return x, E, psi, V


st.set_page_config(page_title="势垒本征态求解", layout="wide")

st.title("一维势垒中的能量本征态求解")

st.sidebar.header("势垒参数")

x1 = st.sidebar.slider("左势垒起点", -8.0, 8.0, -3.0, 0.1)
x2 = st.sidebar.slider("左势垒终点", -8.0, 8.0, -2.0, 0.1)
x3 = st.sidebar.slider("右势垒起点", -8.0, 8.0, 2.0, 0.1)
x4 = st.sidebar.slider("右势垒终点", -8.0, 8.0, 3.0, 0.1)

V1 = st.sidebar.slider("左势垒高度 / eV", -3.0, 5.0, 1.0, 0.1)
V2 = st.sidebar.slider("右势垒高度 / eV", -3.0, 5.0, 1.0, 0.1)

num_states = st.sidebar.slider("计算能级数", 5, 40, 21, 1)
N = st.sidebar.slider("空间网格数 N", 300, 2000, 800, 100)

if not (x1 < x2 < x3 < x4):
    st.error("势垒位置需要满足：左起点 < 左终点 < 右起点 < 右终点")
    st.stop()

x, E, psi, V = solver_run(
    Vx0=(x1, x2, x3, x4),
    V_val=(V1, V2),
    num_states=num_states,
    N=N
)

state = st.selectbox(
    "选择能级编号",
    list(range(num_states)),
    format_func=lambda n: f"n = {n}, E = {E[n]:.4f} eV"
)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("前几个能级")
    for i in range(num_states):
        st.write(f"n = {i:2d}    E = {E[i]:.6f} eV")

with col2:
    st.subheader(f"第 {state} 个能量本征态")

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(x, np.real(psi[:, state]), label=f"Re ψ{state}")
    ax1.plot(x, np.imag(psi[:, state]), label=f"Im ψ{state}")
    ax1.plot(x, np.abs(psi[:, state]) ** 2, label=f"|ψ{state}|²")
    ax1.set_xlabel("x / nm")
    ax1.set_ylabel("wave function / probability density")
    y_max = max(
        np.max(np.real(psi[:, state])),
        np.max(np.imag(psi[:, state])),
        np.max(np.abs(psi[:, state])**2)
    )
    ax1.set_ylim(-1.2*y_max,1.2*y_max)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x, V, color="black", linewidth=2, label="V(x)")
    ax2.axhline(E[state], color="gray", linestyle="--", linewidth=1, label=f"E{state}")
    ax2.set_ylabel("Energy / eV")
    ax2.set_ylim(-3,3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    st.pyplot(fig)

st.caption("当前版本只展示定态本征函数，不处理叠加态和含时演化。")