import numpy as np

def spatial_inertia(m, I_com, c):
    """生成6x6空间惯性矩阵 (Featherstone [ω;v] 约定)"""
    cx = np.array([
        [0,    -c[2],  c[1]],
        [c[2],  0,    -c[0]],
        [-c[1], c[0],  0   ]
    ])
    I_rot = I_com + m * (cx @ cx.T)
    return np.block([
        [I_rot,   m * cx],
        [m * cx.T,  m * np.eye(3)]
    ])

m = 3.5
I = np.array([[2.0, 0.3, 0.1],
                   [0.3, 4.0, 0.2],
                   [0.1, 0.2, 5.0]])
c = np.array([0.15, -0.08, 0.22])

np.random.seed(42)
w = np.random.randn(3)
v = np.random.randn(3)
dw = np.random.randn(3)
dv = np.random.randn(3)


I_6 = spatial_inertia(m, I, c)
v_6 = np.concatenate([w, v])
a_6 = np.concatenate([dw, dv])
f6 = I_6 @ a_6
k6 = 0.5 * v_6 @ I_6 @ v_6

v_c = v + np.cross(w, c)
a_c = dv + np.cross(dw, c)
F3 = m * a_c
N3 = I @ dw + m * np.cross(c, a_c)
f3 = np.concatenate([N3, F3])
k3 = 0.5 * w @ I @ w + 0.5 * m * (v_c @ v_c)

print(f6)
print(f3)
print(k6)
print(k3)