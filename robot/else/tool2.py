import numpy as np
from math import pi

def rot_to_mat(r):
    """
    旋转向量转旋转矩阵
    """
    r = np.asarray(r, dtype=float)
    theta = np.linalg.norm(r)

    if theta < 1e-12:
        return np.eye(3)

    k = r / theta  # 单位轴
    x, y, z = k

    K = np.array([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])

    I = np.eye(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def mat_to_rot(R):
    """
    旋转矩阵转旋转向量
    """
    R = np.asarray(R, dtype=float)

    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # 角度接近 0
    if theta < 1e-10:
        return np.zeros(3)

    # 角度接近 pi
    if np.isclose(theta, np.pi, atol=1e-10):
        A = (R + np.eye(3)) / 2.0
        idx = int(np.argmax(np.diag(A)))
        axis = A[:, idx]
        axis = axis / np.linalg.norm(axis)
        return axis * theta

    # 一般情况
    v = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    axis = v / (2.0 * np.sin(theta))
    axis = axis / np.linalg.norm(axis)
    return axis * theta


def skew(in_):
    """
    skew 在三维向量和 3x3 反对称矩阵之间转换
    S = skew(v) 和 v = skew(A)
    """
    in_ = np.asarray(in_)
    
    if in_.shape == (3, 3):          # 矩阵转向量：v = skew(A)
        out = 0.5 * np.array([
            [in_[2, 1] - in_[1, 2]],
            [in_[0, 2] - in_[2, 0]],
            [in_[1, 0] - in_[0, 1]]
        ])
    else:                            # 向量转矩阵：S = skew(v)
        v = in_.flatten()
        out = np.array([
            [ 0,       -v[2],  v[1]],
            [ v[2],     0,    -v[0]],
            [-v[1],     v[0],  0   ]
        ])
    return out


def crm(v):
    """
    空间叉乘算子（运动）
    crm(v)*m 为运动向量 v 与 m 的叉积
    """
    v = np.asarray(v, dtype=float).flatten()
    vcross = np.array([
        [ 0,    -v[2],  v[1],   0,     0,     0   ],
        [ v[2],  0,    -v[0],   0,     0,     0   ],
        [-v[1],  v[0],  0,      0,     0,     0   ],
        [ 0,    -v[5],  v[4],   0,    -v[2],  v[1]],
        [ v[5],  0,    -v[3],   v[2],  0,    -v[0]],
        [-v[4],  v[3],  0,     -v[1],  v[0],  0   ]
    ])
    return vcross


def crf(v):
    """
    空间叉乘算子（力）
    crf(v)*f 为运动向量 v 与力向量 f 的叉积
    """
    return -crm(v).T


def Fpt(fp, p):
    """
    点上的力 --> 空间/平面力
    f = Fpt(fp, p) 将作用在点 p 上的线性力 fp 转换为等效空间或平面力
    """
    fp = np.asarray(fp, dtype=float)
    p = np.asarray(p, dtype=float)
    f = np.vstack((np.cross(p, fp, axis=0), fp))
    return f


def mcI(i1, i2=None, i3=None):
    """
    刚体惯量 <--> 质量、质心和转动惯量
    rbi = mcI(m, c, I) 和 [m, c, I] = mcI(rbi)
    """
    if i2 is None and i3 is None:
        return rbi_to_mcI(i1)
    else:
        return mcI_to_rbi(i1, i2, i3)


def mcI_to_rbi(m, c, I):
    """质量、质心、转动惯量转刚体惯量"""
    c = np.asarray(c, dtype=float).flatten()

    if len(c) == 3:                      # 空间
        C = skew(c)
        rbi = np.block([
            [I + m*C@C.T,  m*C],
            [m*C.T,        m*np.eye(3)]
        ])
    else:                                # 平面
        rbi = np.array([
            [ I + m*np.dot(c, c), -m*c[1],  m*c[0]],
            [-m*c[1],              m,       0     ],
            [ m*c[0],              0,       m     ]
        ])
    return rbi


def rbi_to_mcI(rbi):
    """刚体惯量转质量、质心、转动惯量"""
    rbi = np.asarray(rbi, dtype=float)

    if rbi.shape == (6, 6):              # 空间
        m = rbi[5, 5]
        mC = rbi[0:3, 3:6]
        c = skew(mC)/m
        I = rbi[0:3, 0:3] - mC@mC.T/m
    else:                                # 平面
        m = rbi[2, 2]
        c = np.array([rbi[2, 0], -rbi[1, 0]])/m
        I = rbi[0, 0] - m*np.dot(c, c)
    return m, c, I



def pluho(in_):
    """
    Plucker <--> 4x4 齐次坐标变换
    X = pluho(T) 和 T = pluho(X)
    """
    in_ = np.asarray(in_, dtype=float)

    if in_.shape == (6, 6):              # Plucker -> 4x4 齐次
        E = in_[0:3, 0:3]
        mErx = in_[3:6, 0:3]             # -E r 的叉乘
        out = np.block([
            [E,               skew(mErx@E.T)],
            [np.zeros((1, 3)), np.array([[1]])]
        ])
    else:                                # 4x4 齐次 -> Plucker
        E = in_[0:3, 0:3]
        mEr = in_[0:3, 3]                # -E r
        out = np.block([
            [E,               np.zeros((3, 3))],
            [skew(mEr)@E,     E]
        ])
    return out



def plux(i1, i2=None):
    """
    Plucker 坐标变换的合成/分解
    X = plux(E, r) 和 [E, r] = plux(X)
    """
    if i2 is not None:                   # E, r --> X
        E = np.asarray(i1, dtype=float)
        r = np.asarray(i2, dtype=float).flatten()

        X = np.block([
            [E,             np.zeros((3, 3))],
            [-E@skew(r),    E]
        ])
        return X

    else:                                # X --> E, r
        X = np.asarray(i1, dtype=float)
        E = X[0:3, 0:3]
        r = -skew(E.T @ X[3:6, 0:3])
        return E, r



def rotx(theta):
    """
    空间坐标变换（绕 X 轴旋转）
    """
    c = np.cos(theta)
    s = np.sin(theta)

    X = np.array([
        [1,  0,  0,  0,  0,  0],
        [0,  c,  s,  0,  0,  0],
        [0, -s,  c,  0,  0,  0],
        [0,  0,  0,  1,  0,  0],
        [0,  0,  0,  0,  c,  s],
        [0,  0,  0,  0, -s,  c]
    ])
    return X



def roty(theta):
    """
    空间坐标变换（绕 Y 轴旋转）
    """
    c = np.cos(theta)
    s = np.sin(theta)

    X = np.array([
        [c,  0, -s,  0,  0,  0],
        [0,  1,  0,  0,  0,  0],
        [s,  0,  c,  0,  0,  0],
        [0,  0,  0,  c,  0, -s],
        [0,  0,  0,  0,  1,  0],
        [0,  0,  0,  s,  0,  c]
    ])
    return X



def rotz(theta):
    """
    空间坐标变换（绕 Z 轴旋转）
    """
    c = np.cos(theta)
    s = np.sin(theta)

    X = np.array([
        [ c,  s,  0,  0,  0,  0],
        [-s,  c,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  c,  s,  0],
        [ 0,  0,  0, -s,  c,  0],
        [ 0,  0,  0,  0,  0,  1]
    ])
    return X



def Vpt(v, p):
    """
    空间/平面速度 --> 各点的速度
    vp = Vpt(v, p) 计算空间/平面速度 v 在点 p 处产生的线速度 vp
    """
    v = np.asarray(v, dtype=float)
    p = np.asarray(p, dtype=float)

    # 保证 v 为二维（列向量也视为 n×1）
    if v.ndim == 1:
        v = v.reshape(-1, 1)

    if v.shape[1] == 1 and p.shape[1] > 1:
        v = np.tile(v, (1, p.shape[1]))

    if v.shape[0] == 6:                  # 3D 点与速度
        vp = v[3:6, :] + np.cross(v[0:3, :], p, axis=0)
    else:                                # 2D 点与速度
        vp = v[1:3, :] + np.vstack((-v[0, :]*p[1, :], v[0, :]*p[0, :]))
    return vp



def xlt(r):
    """
    空间坐标变换（原点平移），r 为 3D 向量（行或列均可）
    """
    r = np.asarray(r, dtype=float).flatten()

    X = np.array([
        [ 1,     0,     0,    0,  0,  0],
        [ 0,     1,     0,    0,  0,  0],
        [ 0,     0,     1,    0,  0,  0],
        [ 0,     r[2], -r[1], 1,  0,  0],
        [-r[2],  0,     r[0], 0,  1,  0],
        [ r[1], -r[0],  0,    0,  0,  1]
    ])
    return X



def Xpt(X, p):
    """
    将 Plucker/平面坐标变换 X 应用于 2D/3D 点 p
    """
    X = np.asarray(X, dtype=float)
    p = np.asarray(p, dtype=float)

    if X.shape == (6, 6):                # 3D 点
        E = X[0:3, 0:3]
        r = -skew(E.T @ X[3:6, 0:3])
    else:                                # 2D 点
        E = X[1:3, 1:3]
        r = np.array([
            X[1, 2]*X[1, 0] + X[2, 2]*X[2, 0],
            X[1, 2]*X[2, 0] - X[2, 2]*X[1, 0]
        ])

    if p.ndim == 1:
        p = p.reshape(-1, 1)

    if p.shape[1] > 1:
        r = np.tile(r, (1, p.shape[1]))

    xp = E @ (p - r)
    return xp



def XtoV(X):
    """
    从小角度变换中提取空间/平面向量
    """
    X = np.asarray(X, dtype=float)

    if X.shape == (6, 6):                # Plucker 变换 -> 空间向量
        v = 0.5 * np.array([
            X[1, 2] - X[2, 1],
            X[2, 0] - X[0, 2],
            X[0, 1] - X[1, 0],
            X[4, 2] - X[5, 1],
            X[5, 0] - X[3, 2],
            X[3, 1] - X[4, 0]
        ])
    else:                                # 平面变换 -> 平面向量
        v = np.array([
            X[1, 2],
            (X[2, 0] + X[1, 1]*X[2, 0] + X[1, 2]*X[1, 0])/2,
            (-X[1, 0] - X[1, 1]*X[1, 0] + X[1, 2]*X[2, 0])/2
        ])
    return v


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=10, linewidth=200)
    r = np.random.uniform(-1, 1, 3)
    t = np.random.uniform(-1, 1, 3)
    R = rot_to_mat(r)
    T = np.block([
        [R,                t.reshape(3, 1)],
        [np.zeros((1, 3)), np.array([[1]])]
    ])

    print("T:\n", T)

    X1 = pluho(T)
    print("X1:\n", X1)
    T1 = pluho(X1)
    print("T1:\n", T1)



    X2 = plux(R, t)
    print("X2:\n", X2)
    R2,t2 = plux(X2)
    print("R2:\n", R2)
    print("t2:\n", t2)

    

    m = 3.5
    c = np.array([0.15, -0.08, 0.22])
    I = np.array([[2.0, 0.3, 0.1],
                [0.3, 4.0, 0.2],
                [0.1, 0.2, 5.0]])

    I6 = mcI(m, c, I)
    print(I6)
    m2,c2,I2 = mcI(I6)
    print(m)
    print(m2)
    print(c)
    print(c2)
    print(I)
    print(I2)


    