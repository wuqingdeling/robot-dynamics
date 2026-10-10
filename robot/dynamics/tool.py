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


def mcI_to_rbi(m, c, I):
    """质量、质心、转动惯量转刚体惯量"""
    c = np.asarray(c, dtype=float).flatten()
    C = skew(c)
    rbi = np.block([
        [I + m*C@C.T,  m*C],
        [m*C.T,        m*np.eye(3)]
    ])
    return rbi


def rz(theta):
    """
    齐次变换（绕 Z 轴旋转）
    """
    c = cos(theta)
    s = sin(theta)
    T = np.array([
        [c, -s, 0, 0], 
        [s, c, 0, 0], 
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    return T


def rotz(theta):
    """
    空间坐标变换（绕 Z 轴旋转）
    """
    c = cos(theta)
    s = sin(theta)
    X = np.array([
        [ c,  s,  0,  0,  0,  0],
        [-s,  c,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  c,  s,  0],
        [ 0,  0,  0, -s,  c,  0],
        [ 0,  0,  0,  0,  0,  1]
    ])
    return X


def plux(R, P):
    """从 4x4 齐次变换矩阵 T 构造 6x6 空间变换矩阵"""
    R = np.asarray(R)
    P = np.asarray(P).reshape(3)
    return np.block([
        [R,               np.zeros((3, 3))],
        [-R @ skew(P),    R],
    ])


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