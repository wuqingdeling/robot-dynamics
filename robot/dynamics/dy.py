import numpy as np

def dynamic_antisymmetric(v):
    v = np.asarray(v, dtype=float)
    if v.size != 3:
        raise ValueError("Wrong number of input arguments")

    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]])


def dynamic_bianhuanjuzhen(i, theta, j):
    c, s = np.cos(theta), np.sin(theta)

    if i == 1:
        m = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])
    elif i in (2, 3, 5, 7):
        m = np.array([[c, -s, 0],
                      [0,  0, 1],
                      [-s, -c, 0]])
    elif i in (4, 6):
        m = np.array([[c, -s,  0],
                      [0,  0, -1],
                      [s,  c,  0]])
    elif i == 8:
        m = np.zeros((3, 3))
    else:
        raise ValueError(f"unsupported i={i}")

    if j == 1:
        return m.T
    else:
        return m

def dynamic_model1008_02(theta):
    theta = np.asarray(theta, dtype=float)

    p = 1e-3 * np.array([
        [0,   0,   0,   0,   0,   0,   0],
        [0, 250,   0, 150,   0, 200,   0],
        [150, 0, 150,   0, 135,   0, 100],
    ], dtype=float)

    rc = 1e-3 * np.array([
        [0,   0,   0,   0,   0,   0,   0],
        [0, 125,   0,  75,   0, 100,   0],
        [75,  0,  75,   0,  75,   0,  50],
    ], dtype=float)

    m = np.array([5, 18, 10, 12, 8.6, 16, 10], dtype=float)
    Ic = []

    Icrba = np.zeros((3, 24))
    mc = np.zeros(8)

    for i in range(6, -1, -1):          # i = 7..1 -> 0..6
        mc[i] = m[i] + mc[i + 1]

    cj = np.zeros((3, 8))
    for i in range(6, -1, -1):
        rm2 = dynamic_bianhuanjuzhen(i + 1, theta[i + 1], 0)
        cj[:, i] = (m[i] * (p[:, i] + rc[:, i]) +
                    mc[i + 1] * (p[:, i] + rm2 @ cj[:, i + 1])) / mc[i]

    for i in range(6, -1, -1):
        rm = dynamic_bianhuanjuzhen(i, theta[i], 1)
        rm1 = dynamic_bianhuanjuzhen(i + 1, theta[i + 1], 0)

        a = p[:, i] + rc[:, i] - cj[:, i]
        aa = np.outer(a, a) * np.eye(3) - np.outer(a, a)

        b = p[:, i] + rm1 @ cj[:, i + 1] - cj[:, i]
        bb = np.outer(b, b) * np.eye(3) - np.outer(b, b)

        Icrba[:, 3 * i:3 * i + 3] = (
            Ic[:, 3 * i:3 * i + 3]
            + m[i] * aa
            + rm1 @ Icrba[:, 3 * i + 1:3 * i + 4] @ rm1.T
            + mc[i + 1] * bb
        )

    f = np.zeros((3, 7))
    n = np.zeros((3, 7))
    Hq = np.zeros((7, 7))
    z0 = np.array([0, 0, 1], dtype=float)

    for j in range(6, -1, -1):
        for i in range(j, -1, -1):
            rm = dynamic_bianhuanjuzhen(i, theta[i], 1)

            if i == j:
                antis_cj = dynamic_antisymmetric(cj[:, i])
                f[:, i] = mc[i] * (-antis_cj @ (rm @ z0))
                n[:, i] = Icrba[:, 3 * i:3 * i + 3] @ (rm @ z0) + antis_cj @ f[:, i]
                Hq[i, j] = (rm @ z0) @ n[:, i]
            elif i < j:
                rm1 = dynamic_bianhuanjuzhen(i + 1, theta[i + 1], 0)
                f[:, i] = rm1 @ f[:, i + 1]
                antis_p = dynamic_antisymmetric(p[:, i])
                n[:, i] = rm1 @ n[:, i + 1] + antis_p @ f[:, i]
                Hq[i, j] = (rm @ z0) @ n[:, i]

    for i in range(1, 7):
        for j in range(i - 1, -1, -1):
            Hq[i, j] = Hq[j, i]

    return Hq