from math import sin, cos, pi
import numpy as np

axis1 = np.array([[0], [0], [1]])
axis2 = np.array([[0], [0], [1]])
axis3 = np.array([[0], [0], [1]])
axis4 = np.array([[0], [0], [1]])
axis5 = np.array([[0], [0], [1]])
axis6 = np.array([[0], [0], [1]])

alpha = np.array([0, -pi/2, 0, pi/2, -pi/2, pi/2]) # 连杆扭转角
a = np.array([0, 0, 0.4, 0, 0, 0]) # 连杆长度
d = np.array([0, 0, 0.02, 0.45, 0, 0.07]) # 连杆偏距
theta = np.array([0, -pi/2, pi/2, 0, 0, 0]) # 关节角

def MDHTrans(alpha, a, d, theta):
    """返回改进DH参数的齐次变换矩阵"""
    T = []
    for i in range(0, 6):
        T.append(np.array([
            [cos(theta[i]), -sin(theta[i]), 0, a[i]],
            [sin(theta[i])*cos(alpha[i]), cos(theta[i])*cos(alpha[i]), -sin(alpha[i]), -d[i]*sin(alpha[i])],
            [sin(theta[i])*sin(alpha[i]), cos(theta[i])*sin(alpha[i]), cos(alpha[i]), d[i]*cos(alpha[i])],
            [0, 0, 0, 1]
        ]))
    return T

def ccross(a, b):
    # 确保输入是 (3, 1) 
    result = np.array([
        [a[1, 0] * b[2, 0] - a[2, 0] * b[1, 0]],
        [a[2, 0] * b[0, 0] - a[0, 0] * b[2, 0]],
        [a[0, 0] * b[1, 0] - a[1, 0] * b[0, 0]]
    ])
    return result

# 调用运动学参数计算函数
def motion_para_clc(axis, R_inv, Po, dq, ddq, Pc, w_pre, v_pre, dw_pre, dv_pre):
    w = R_inv @ w_pre + dq * axis
    v = R_inv @ (v_pre + ccross(w_pre, Po))
    dw = R_inv @ dw_pre + ccross(R_inv @ w_pre, dq * axis) + ddq * axis
    dv = R_inv @ (dv_pre + ccross(dw_pre, Po) + ccross(w_pre, ccross(w_pre, Po)))
    dvc = dv + ccross(dw, Pc) + ccross(w, ccross(w, Pc))
    return w, v, dw, dv, dvc

def fn_clc(R, Po, w, dw, dv, dvc, m, Pc, I, f_next, n_next):
    f = R @ f_next + m * dvc
    n = R @ n_next + ccross(Po, R @ f_next) + I @ dw + ccross(w, I @ w) + m * ccross(Pc, dv)
    return f, n

def calcAcceleration(q, dq, ddq, g=0, x=(0,0,0,0,0,0), y=(0,0,0,0,0,0), z=(0,0,0,0,0,0)):
    q1, q2, q3, q4, q5, q6 = q
    dq1, dq2, dq3, dq4, dq5, dq6 = dq
    ddq1, ddq2, ddq3, ddq4, ddq5, ddq6 = ddq
    x1, x2, x3, x4, x5, x6 = x
    y1, y2, y3, y4, y5, y6 = y
    z1, z2, z3, z4, z5, z6 = z

    # 坐标系变换矩阵
    T_list = MDHTrans(alpha, a, d, theta + q)
    T_list.append(np.eye(4))
    P = [T[0:3, 3:4] for T in T_list]
    R = [T[0:3, 0:3] for T in T_list]
    R_inv = [R_mat.T for R_mat in R]
    (P01, P12, P23, P34, P45, P56, P67) = P
    (R01, R12, R23, R34, R45, R56, R67) = R
    (R10, R21, R32, R43, R54, R65, R76) = R_inv

    w00 = np.array([[0], [0], [0]])
    v00 = np.array([[0], [0], [0]])
    dw00 = np.array([[0], [0], [0]])
    dv00 = np.array([[0], [0], [g]])

    Pc11 = np.array([[x1], [y1], [z1]])
    Pc22 = np.array([[x2], [y2], [z2]])
    Pc33 = np.array([[x3], [y3], [z3]])
    Pc44 = np.array([[x4], [y4], [z4]])
    Pc55 = np.array([[x5], [y5], [z5]])
    Pc66 = np.array([[x6], [y6], [z6]])

    # 外推：连杆角速度、角加速度、线加速度、质心加速度、质心受力
    w11, v11, dw11, dv11, dvc11 = motion_para_clc(axis1, R10, P01, dq1, ddq1, Pc11, w00, v00, dw00, dv00)
    w22, v22, dw22, dv22, dvc22 = motion_para_clc(axis2, R21, P12, dq2, ddq2, Pc22, w11, v11, dw11, dv11)
    w33, v33, dw33, dv33, dvc33 = motion_para_clc(axis3, R32, P23, dq3, ddq3, Pc33, w22, v22, dw22, dv22)
    w44, v44, dw44, dv44, dvc44 = motion_para_clc(axis4, R43, P34, dq4, ddq4, Pc44, w33, v33, dw33, dv33)
    w55, v55, dw55, dv55, dvc55 = motion_para_clc(axis5, R54, P45, dq5, ddq5, Pc55, w44, v44, dw44, dv44)
    w66, v66, dw66, dv66, dvc66 = motion_para_clc(axis6, R65, P56, dq6, ddq6, Pc66, w55, v55, dw55, dv55)
    return w66, v66, dw66, dv66

def calcForce(q, dq, ddq, g=0, x=(0,0,0,0,0,0), y=(0,0,0,0,0,0), z=(0,0,0,0,0,0), m=(0,0,0,0,0,0), I1=(0,0,0,0,0,0), I2=(0,0,0,0,0,0), I3=(0,0,0,0,0,0), I4=(0,0,0,0,0,0), I5=(0,0,0,0,0,0), I6=(0,0,0,0,0,0)):
    q1, q2, q3, q4, q5, q6 = q
    dq1, dq2, dq3, dq4, dq5, dq6 = dq
    ddq1, ddq2, ddq3, ddq4, ddq5, ddq6 = ddq
    m1, m2, m3, m4, m5, m6 = m
    x1, x2, x3, x4, x5, x6 = x
    y1, y2, y3, y4, y5, y6 = y
    z1, z2, z3, z4, z5, z6 = z
    Ixx1, Iyy1, Izz1, Ixy1, Ixz1, Iyz1 = I1
    Ixx2, Iyy2, Izz2, Ixy2, Ixz2, Iyz2 = I2
    Ixx3, Iyy3, Izz3, Ixy3, Ixz3, Iyz3 = I3
    Ixx4, Iyy4, Izz4, Ixy4, Ixz4, Iyz4 = I4
    Ixx5, Iyy5, Izz5, Ixy5, Ixz5, Iyz5 = I5
    Ixx6, Iyy6, Izz6, Ixy6, Ixz6, Iyz6 = I6

    # 坐标系变换矩阵
    T_list = MDHTrans(alpha, a, d, theta + q)
    T_list.append(np.eye(4))
    P = [T[0:3, 3:4] for T in T_list]
    R = [T[0:3, 0:3] for T in T_list]
    R_inv = [R_mat.T for R_mat in R]
    (P01, P12, P23, P34, P45, P56, P67) = P
    (R01, R12, R23, R34, R45, R56, R67) = R
    (R10, R21, R32, R43, R54, R65, R76) = R_inv

    w00 = np.array([[0], [0], [0]])
    v00 = np.array([[0], [0], [0]])
    dw00 = np.array([[0], [0], [0]])
    dv00 = np.array([[0], [0], [g]])

    Pc11 = np.array([[x1], [y1], [z1]])
    Pc22 = np.array([[x2], [y2], [z2]])
    Pc33 = np.array([[x3], [y3], [z3]])
    Pc44 = np.array([[x4], [y4], [z4]])
    Pc55 = np.array([[x5], [y5], [z5]])
    Pc66 = np.array([[x6], [y6], [z6]])

    # 外推：连杆角速度、角加速度、线加速度、质心加速度、质心受力
    w11, v11, dw11, dv11, dvc11 = motion_para_clc(axis1, R10, P01, dq1, ddq1, Pc11, w00, v00, dw00, dv00)
    w22, v22, dw22, dv22, dvc22 = motion_para_clc(axis2, R21, P12, dq2, ddq2, Pc22, w11, v11, dw11, dv11)
    w33, v33, dw33, dv33, dvc33 = motion_para_clc(axis3, R32, P23, dq3, ddq3, Pc33, w22, v22, dw22, dv22)
    w44, v44, dw44, dv44, dvc44 = motion_para_clc(axis4, R43, P34, dq4, ddq4, Pc44, w33, v33, dw33, dv33)
    w55, v55, dw55, dv55, dvc55 = motion_para_clc(axis5, R54, P45, dq5, ddq5, Pc55, w44, v44, dw44, dv44)
    w66, v66, dw66, dv66, dvc66 = motion_para_clc(axis6, R65, P56, dq6, ddq6, Pc66, w55, v55, dw55, dv55)

    Ic1 = np.array([
        [Ixx1, -Ixy1, -Ixz1],
        [-Ixy1, Iyy1, -Iyz1],
        [-Ixz1, -Iyz1, Izz1]
    ])
    Ic2 = np.array([
        [Ixx2, -Ixy2, -Ixz2],
        [-Ixy2, Iyy2, -Iyz2],
        [-Ixz2, -Iyz2, Izz2]
    ])
    Ic3 = np.array([
        [Ixx3, -Ixy3, -Ixz3],
        [-Ixy3, Iyy3, -Iyz3],
        [-Ixz3, -Iyz3, Izz3]
    ])
    Ic4 = np.array([
        [Ixx4, -Ixy4, -Ixz4],
        [-Ixy4, Iyy4, -Iyz4],
        [-Ixz4, -Iyz4, Izz4]
    ])
    Ic5 = np.array([
        [Ixx5, -Ixy5, -Ixz5],
        [-Ixy5, Iyy5, -Iyz5],
        [-Ixz5, -Iyz5, Izz5]
    ])
    Ic6 = np.array([
        [Ixx6, -Ixy6, -Ixz6],
        [-Ixy6, Iyy6, -Iyz6],
        [-Ixz6, -Iyz6, Izz6]
    ])

    II1 = Ic1 + m1*((Pc11.T @ Pc11) * np.eye(3) - Pc11 @ Pc11.T)
    II2 = Ic2 + m2*((Pc22.T @ Pc22) * np.eye(3) - Pc22 @ Pc22.T)
    II3 = Ic3 + m3*((Pc33.T @ Pc33) * np.eye(3) - Pc33 @ Pc33.T)
    II4 = Ic4 + m4*((Pc44.T @ Pc44) * np.eye(3) - Pc44 @ Pc44.T)
    II5 = Ic5 + m5*((Pc55.T @ Pc55) * np.eye(3) - Pc55 @ Pc55.T)
    II6 = Ic6 + m6*((Pc66.T @ Pc66) * np.eye(3) - Pc66 @ Pc66.T)

    f77 = np.array([[0], [0], [0]])
    n77 = np.array([[0], [0], [0]])
    
    f66, n66 = fn_clc(R67, P67, w66, dw66, dv66, dvc66, m6, Pc66, II6, f77, n77)
    f55, n55 = fn_clc(R56, P56, w55, dw55, dv55, dvc55, m5, Pc55, II5, f66, n66)
    f44, n44 = fn_clc(R45, P45, w44, dw44, dv44, dvc44, m4, Pc44, II4, f55, n55)
    f33, n33 = fn_clc(R34, P34, w33, dw33, dv33, dvc33, m3, Pc33, II3, f44, n44)
    f22, n22 = fn_clc(R23, P23, w22, dw22, dv22, dvc22, m2, Pc22, II2, f33, n33)
    f11, n11 = fn_clc(R12, P12, w11, dw11, dv11, dvc11, m1, Pc11, II1, f22, n22)

    tau = np.array([[n11[2,0]], [n22[2,0]], [n33[2,0]], [n44[2,0]], [n55[2,0]], [n66[2,0]]])
    return tau

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=10, linewidth=200)
    q = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    dq = np.array([0.11, 0.11, 0.11, 0.11, 0.11, 0.11])
    ddq = np.array([0, 0, 0, 0, 0, 0])
    w, v, dw, dv = calcAcceleration(q, dq, ddq)
    print("v6:\n", v)
    print("w6:\n", w)
    print("dv6:\n", dv)
    print("dw6:\n", dw)
    
    m = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    x = np.array([0, 0, 0, 0, 0, 0])
    y = np.array([0, 0, 0, 0, 0, 0])
    z = np.array([0, 0, 0, 0, 0, 0])
    I1 = np.array([0.1, 0.1, 0.1, 0, 0, 0])
    I2 = np.array([0.1, 0.1, 0.1, 0, 0, 0])
    I3 = np.array([0.1, 0.1, 0.1, 0, 0, 0])
    I4 = np.array([0.1, 0.1, 0.1, 0, 0, 0])
    I5 = np.array([0.1, 0.1, 0.1, 0, 0, 0])
    I6 = np.array([0.1, 0.1, 0.1, 0, 0, 0])
    tau = calcForce(q, dq, ddq)
    print("tau:\n", tau)