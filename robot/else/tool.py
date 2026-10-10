from math import sin, cos, pi
import numpy as np


def ccross(a, b):
    """向量叉乘"""
    result = np.array([
        [a[1, 0] * b[2, 0] - a[2, 0] * b[1, 0]],
        [a[2, 0] * b[0, 0] - a[0, 0] * b[2, 0]],
        [a[0, 0] * b[1, 0] - a[1, 0] * b[0, 0]]
    ])
    return result


def skew(w):
    w = np.asarray(w).reshape(3)   # (3,1) 或 (3,) 都变成 (3,)
    return np.array([
        [0,      -w[2],  w[1]],
        [w[2],    0,    -w[0]],
        [-w[1],  w[0],    0   ]
    ])


def inv_SE3(T):
    """SE(3) 逆"""
    R = T[:3, :3]
    p = T[:3, 3:4]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3:4] = -R.T @ p
    return Tinv


def AD_SE3(T):
    """SE(3) 的伴随变换 6x6"""
    R = T[:3, :3]
    p = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = skew(p) @ R
    return Ad


def ad_screw(S):
    """
    计算螺旋轴 S 的小伴随矩阵 ad_S (6x6)

    参数:
        S : (6,) 或 (6,1) 螺旋轴 [ω; v]
    返回:
        ads : (6,6) 小伴随矩阵
              [[ω̂,  0 ],
               [v̂,  ω̂ ]]
    """
    S = np.asarray(S).reshape(6)
    w = S[:3]      # (3,)
    w = S[:3]      # (3,)
    v = S[3:]      # (3,)

    W = skew(w)  # (3,3)
    V = skew(v)  # (3,3)
    Z = np.zeros((3, 3))

    ads = np.zeros((6, 6))
    ads[:3, :3] = W
    ads[:3, 3:] = Z
    ads[3:, :3] = V
    ads[3:, 3:] = W
    return ads


def build_X(R, P):
    """从 3x3 旋转矩阵 R 和 3x1 平移向量 P 构造 6x6 空间变换矩阵"""
    x, y, z = P.flatten()
    P_cross = np.array([
        [0, -z,  y],
        [z,  0, -x],
        [-y,  x,  0]
    ])
    X = np.block([
        [R, np.zeros((3, 3))],
        [-R @ P_cross, R]
    ])
    return X


def exp_so3_2_SO3(w, theta):
    """
    指数映射 so(3) -> SO(3)
    w    : (3,) 或 (3,1) 旋转轴（应已归一化，若否则内部归一化）
    theta: 旋转角度
    """
    w = np.asarray(w).reshape(3)
    # 若 w 不是单位向量，归一化（角度相应缩放）
    norm_w = np.linalg.norm(w)
    if norm_w > 1e-12:
        w = w / norm_w
        theta = theta * norm_w

    W = skew(w)
    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * (W @ W)
    return R


def exp_twist_2_SE3(S, theta):
    """
    螺旋轴指数运算
    S    : 6x1 或 (6,) 螺旋轴 [w; v]
    theta: 旋转角度
    return: 4x4 SE(3) 齐次变换矩阵
    """
    S = np.asarray(S).reshape(6)
    w = S[:3]              # (3,)
    v = S[3:]              # (3,)

    # 归一化：如果 w 不是单位向量，需要同时缩放 w 和 v，并缩放 theta
    norm_w = np.linalg.norm(w)
    if norm_w > 1e-12:
        w = w / norm_w
        v = v / norm_w
        theta = theta * norm_w

    R = exp_so3_2_SO3(w, theta)

    # G(θ) = Iθ + (1-cosθ)ω̂ + (θ-sinθ)ω̂²
    W = skew(w)
    I = np.eye(3)
    G = I * theta + (1 - np.cos(theta)) * W + (theta - np.sin(theta)) * (W @ W)

    # 组装 SE(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:4] = G @ v.reshape(3, 1)
    return T


def spatial_cross(v, m):
    """
    实现空间叉乘 v x m (公式 2.33)
    v = [w; v_O], m = [m; m_O]
    """
    
    w = v[0:3]
    v_O = v[3:6]
    m_top = m[0:3]
    m_bot = m[3:6]

    # 构造 3x3 叉乘矩阵
    def skew(vec):
        x, y, z = vec.flatten()
        return np.array([[0, -z, y], 
                         [z, 0, -x], 
                         [-y, x, 0]])

    w_skew = skew(w)
    v_skew = skew(v_O)

    # 根据公式 2.33:
    # top = w x m
    # bot = w x m_O + v_O x m
    top = w_skew @ m_top
    bot = w_skew @ m_bot + v_skew @ m_top
    return np.vstack((top, bot))


def motion_clc(R_inv, P, dq, ddq, w_pre, v_pre, dw_pre, dv_pre, Pc, axis):
    w = R_inv @ w_pre + dq * axis
    v = R_inv @ (v_pre + ccross(w_pre, P))
    dw = R_inv @ dw_pre + ccross(R_inv @ w_pre, dq * axis) + ddq * axis
    dv = R_inv @ (dv_pre + ccross(dw_pre, P) + ccross(w_pre, ccross(w_pre, P)))
    dvc = dv + ccross(dw, Pc) + ccross(w, ccross(w, Pc))
    return w, v, dw, dv, dvc


def FN_clc(w, dw, dvc, m, Ic):
    Fc = m * dvc
    Nc = Ic @ dw + ccross(w, Ic @ w)
    return Fc, Nc


def fn_clc(R, P, Fc, Nc, f_next, n_next, Pc):
    f = R @ f_next + Fc
    n = Nc + R @ n_next + ccross(Pc, Fc) + ccross(P, R @ f_next)
    return f, n