import sympy as sp
from sympy import sin, cos, pi, Matrix, eye, simplify, trigsimp
sp.init_printing()  # 让输出结果更美观

def MDHTrans(alpha, a, d, theta):
    """返回改进DH参数的齐次变换矩阵"""
    return sp.Matrix([
        [cos(theta), -sin(theta), 0, a],
        [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -d*sin(alpha)],
        [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), d*cos(alpha)],
        [0, 0, 0, 1]
    ])

def rc_clc(T_Local, Pc, T_pre):
    T_world = T_pre @ T_Local
    rc = (T_world[0:3, 3:4] + T_world[0:3, 0:3] @ Pc)[2,0]
    return T_world, rc

def motion_para_clc(axis, R_inv, Po, dq, Pc, w_pre, v_pre):
    w = simplify(R_inv @ w_pre + dq * axis)
    # print(f"w = {w}")
    v = simplify(R_inv @ (v_pre + w_pre.cross(Po)))
    # print(f"v = {v}")
    vc = v + simplify(w.cross(Pc))
    # print(f"vc = {vc}")
    return w, v, vc

# 角度 角速度 角加速度
q1, q2, q3, q4, q5, q6 = sp.symbols('q1:7')
dq1, dq2, dq3, dq4, dq5, dq6 = sp.symbols('dq1:7')
ddq1, ddq2, ddq3, ddq4, ddq5, ddq6 = sp.symbols('ddq1:7')
# 质量
m1, m2, m3, m4, m5, m6 = sp.symbols('m1:7')
# 重心
x1, x2, x3, x4, x5, x6 = sp.symbols('x1:7')
y1, y2, y3, y4, y5, y6 = sp.symbols('y1:7')
z1, z2, z3, z4, z5, z6 = sp.symbols('z1:7')
# 惯性张量
Ixx1, Iyy1, Izz1, Ixy1, Ixz1, Iyz1 = sp.symbols('Ixx1 Iyy1 Izz1 Ixy1 Ixz1 Iyz1')
Ixx2, Iyy2, Izz2, Ixy2, Ixz2, Iyz2 = sp.symbols('Ixx2 Iyy2 Izz2 Ixy2 Ixz2 Iyz2')
Ixx3, Iyy3, Izz3, Ixy3, Ixz3, Iyz3 = sp.symbols('Ixx3 Iyy3 Izz3 Ixy3 Ixz3 Iyz3')
Ixx4, Iyy4, Izz4, Ixy4, Ixz4, Iyz4 = sp.symbols('Ixx4 Iyy4 Izz4 Ixy4 Ixz4 Iyz4')
Ixx5, Iyy5, Izz5, Ixy5, Ixz5, Iyz5 = sp.symbols('Ixx5 Iyy5 Izz5 Ixy5 Ixz5 Iyz5')
Ixx6, Iyy6, Izz6, Ixy6, Ixz6, Iyz6 = sp.symbols('Ixx6 Iyy6 Izz6 Ixy6 Ixz6 Iyz6')

Pc11 = Matrix([[x1], [y1], [z1]])
Pc22 = Matrix([[x2], [y2], [z2]])
Pc33 = Matrix([[x3], [y3], [z3]])
Pc44 = Matrix([[x4], [y4], [z4]])
Pc55 = Matrix([[x5], [y5], [z5]])
Pc66 = Matrix([[x6], [y6], [z6]])

Ic1 = Matrix([
    [Ixx1, -Ixy1, -Ixz1],
    [-Ixy1, Iyy1, -Iyz1],
    [-Ixz1, -Iyz1, Izz1]
])
Ic2 = Matrix([
    [Ixx2, -Ixy2, -Ixz2],
    [-Ixy2, Iyy2, -Iyz2],
    [-Ixz2, -Iyz2, Izz2]
])
Ic3 = Matrix([
    [Ixx3, -Ixy3, -Ixz3],
    [-Ixy3, Iyy3, -Iyz3],
    [-Ixz3, -Iyz3, Izz3]
])
Ic4 = Matrix([
    [Ixx4, -Ixy4, -Ixz4],
    [-Ixy4, Iyy4, -Iyz4],
    [-Ixz4, -Iyz4, Izz4]
])
Ic5 = Matrix([
    [Ixx5, -Ixy5, -Ixz5],
    [-Ixy5, Iyy5, -Iyz5],
    [-Ixz5, -Iyz5, Izz5]
])
Ic6 = Matrix([
    [Ixx6, -Ixy6, -Ixz6],
    [-Ixy6, Iyy6, -Iyz6],
    [-Ixz6, -Iyz6, Izz6]
])

axis1 = Matrix([[0], [0], [1]])
axis2 = Matrix([[0], [0], [1]])
axis3 = Matrix([[0], [0], [1]])
axis4 = Matrix([[0], [0], [1]])
axis5 = Matrix([[0], [0], [1]])
axis6 = Matrix([[0], [0], [1]])

# 坐标系变换矩阵
T01 = MDHTrans(    0,   0,    0, q1     )
T12 = MDHTrans(-pi/2,   0,    0, q2-pi/2)
T23 = MDHTrans(    0, 0.4, 0.02, q3+pi/2)
T34 = MDHTrans( pi/2,   0, 0.45, q4     )
T45 = MDHTrans(-pi/2,   0,    0, q5     )
T56 = MDHTrans( pi/2,   0, 0.07, q6     )

# 从齐次变换矩阵提取旋转矩阵和平移向量 注意：Matlab索引从1开始，Python从0开始
T_list = [T01, T12, T23, T34, T45, T56, eye(4)]
P = [T[0:3, 3:4] for T in T_list]
R = [T[0:3, 0:3] for T in T_list]
R_inv = [R_mat.T for R_mat in R]
(P01, P12, P23, P34, P45, P56, P67) = P
(R01, R12, R23, R34, R45, R56, R67) = R
(R10, R21, R32, R43, R54, R65, R76) = R_inv

if False:
    T01, rc11 = rc_clc(T01, Pc11, eye(4))
    T02, rc22 = rc_clc(T12, Pc22, T01)
    T03, rc33 = rc_clc(T23, Pc33, T02)
    T04, rc44 = rc_clc(T34, Pc44, T03)
    T05, rc55 = rc_clc(T45, Pc55, T04)
    T06, rc66 = rc_clc(T56, Pc66, T05)

    for i in range(1, 7):
        idx = f"{i}{i}"
        print(f"rc{idx} = {globals()[f'rc{idx}']}")
else:
    rc11 = z1
    rc22 = x2*cos(q2) - y2*sin(q2)
    rc33 = x3*(-sin(q2)*cos(q3) - sin(q3)*cos(q2)) + y3*(sin(q2)*sin(q3) - cos(q2)*cos(q3)) + 0.4*cos(q2)
    rc44 = x4*(-sin(q2)*cos(q3) - sin(q3)*cos(q2))*cos(q4) - y4*(-sin(q2)*cos(q3) - sin(q3)*cos(q2))*sin(q4) + z4*(-sin(q2)*sin(q3) + cos(q2)*cos(q3)) - 0.45*sin(q2)*sin(q3) + 0.45*cos(q2)*cos(q3) + 0.4*cos(q2)
    rc55 = x5*(-(-sin(q2)*sin(q3) + cos(q2)*cos(q3))*sin(q5) + (-sin(q2)*cos(q3) - sin(q3)*cos(q2))*cos(q4)*cos(q5)) + y5*(-(-sin(q2)*sin(q3) + cos(q2)*cos(q3))*cos(q5) - (-sin(q2)*cos(q3) - sin(q3)*cos(q2))*sin(q5)*cos(q4)) - z5*(-sin(q2)*cos(q3) - sin(q3)*cos(q2))*sin(q4) - 0.45*sin(q2)*sin(q3) + 0.45*cos(q2)*cos(q3) + 0.4*cos(q2)
    rc66 = x6*((-(-sin(q2)*sin(q3) + cos(q2)*cos(q3))*sin(q5) + (-sin(q2)*cos(q3) - sin(q3)*cos(q2))*cos(q4)*cos(q5))*cos(q6) - (-sin(q2)*cos(q3) - sin(q3)*cos(q2))*sin(q4)*sin(q6)) + y6*(-(-(-sin(q2)*sin(q3) + cos(q2)*cos(q3))*sin(q5) + (-sin(q2)*cos(q3) - sin(q3)*cos(q2))*cos(q4)*cos(q5))*sin(q6) - (-sin(q2)*cos(q3) - sin(q3)*cos(q2))*sin(q4)*cos(q6)) + z6*((-sin(q2)*sin(q3) + cos(q2)*cos(q3))*cos(q5) + (-sin(q2)*cos(q3) - sin(q3)*cos(q2))*sin(q5)*cos(q4)) + 0.07*(-sin(q2)*sin(q3) + cos(q2)*cos(q3))*cos(q5) + 0.07*(-sin(q2)*cos(q3) - sin(q3)*cos(q2))*sin(q5)*cos(q4) - 0.45*sin(q2)*sin(q3) + 0.45*cos(q2)*cos(q3) + 0.4*cos(q2)

g = sp.symbols('g')
w00 = Matrix([[0], [0], [0]])
v00 = Matrix([[0], [0], [0]])

if False:
    # 外推：连杆角速度 线速度
    w11, v11, vc11 = motion_para_clc(axis1, R10, P01, dq1, Pc11, w00, v00)
    w22, v22, vc22 = motion_para_clc(axis2, R21, P12, dq2, Pc22, w11, v11)
    w33, v33, vc33 = motion_para_clc(axis3, R32, P23, dq3, Pc33, w22, v22)
    w44, v44, vc44 = motion_para_clc(axis4, R43, P34, dq4, Pc44, w33, v33)
    w55, v55, vc55 = motion_para_clc(axis5, R54, P45, dq5, Pc55, w44, v44)
    w66, v66, vc66 = motion_para_clc(axis6, R65, P56, dq6, Pc66, w55, v55)

    for i in range(1, 7):
        idx = f"{i}{i}"
        print(f"w{idx} = {globals()[f'w{idx}']}")
        print(f"vc{idx} = {globals()[f'vc{idx}']}")
else:
    w11 = Matrix([[0], [0], [dq1]])
    vc11 = Matrix([[-dq1*y1], [dq1*x1], [0]])
    w22 = Matrix([[dq1*cos(q2)], [-dq1*sin(q2)], [dq2]])
    vc22 = Matrix([[-dq1*z2*sin(q2) - dq2*y2], [-dq1*z2*cos(q2) + dq2*x2], [dq1*(x2*sin(q2) + y2*cos(q2))]])
    w33 = Matrix([[-dq1*sin(q2 + q3)], [-dq1*cos(q2 + q3)], [dq2 + dq3]])
    vc33 = Matrix([[-dq1*z3*cos(q2 + q3) - 0.02*dq1*cos(q2 + q3) - dq2*y3 + 0.4*dq2*cos(q3) - dq3*y3], [dq1*z3*sin(q2 + q3) + 0.02*dq1*sin(q2 + q3) - 0.4*dq2*sin(q3) + x3*(dq2 + dq3)], [dq1*(x3*cos(q2 + q3) - y3*sin(q2 + q3)) + 0.4*dq1*sin(q2)]])
    w44 = Matrix([[-dq1*sin(q2 + q3)*cos(q4) + (dq2 + dq3)*sin(q4)], [dq1*sin(q4)*sin(q2 + q3) + (dq2 + dq3)*cos(q4)], [dq1*cos(q2 + q3) + dq4]])
    vc44 = Matrix([[dq1*(0.4*sin(q2) + 0.45*sin(q2 + q3))*sin(q4) - y4*(dq1*cos(q2 + q3) + dq4) + z4*(dq1*sin(q4)*sin(q2 + q3) + (dq2 + dq3)*cos(q4)) + (-0.02*dq1*cos(q2 + q3) + 0.4*dq2*cos(q3) + 0.45*dq2 + 0.45*dq3)*cos(q4)], [dq1*(0.4*sin(q2) + 0.45*sin(q2 + q3))*cos(q4) + x4*(dq1*cos(q2 + q3) + dq4) + z4*(dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4)) - (-0.02*dq1*cos(q2 + q3) + 0.4*dq2*cos(q3) + 0.45*dq2 + 0.45*dq3)*sin(q4)], [-0.02*dq1*sin(q2 + q3) + 0.4*dq2*sin(q3) - x4*(dq1*sin(q4)*sin(q2 + q3) + (dq2 + dq3)*cos(q4)) - y4*(dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))]])
    w55 = Matrix([[-(dq1*cos(q2 + q3) + dq4)*sin(q5) - (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*cos(q5)], [-(dq1*cos(q2 + q3) + dq4)*cos(q5) + (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*sin(q5)], [dq1*sin(q4)*sin(q2 + q3) + dq5 + (dq2 + dq3)*cos(q4)]])
    vc55 = Matrix([[-y5*(dq1*sin(q4)*sin(q2 + q3) + dq5 + (dq2 + dq3)*cos(q4)) - z5*((dq1*cos(q2 + q3) + dq4)*cos(q5) - (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*sin(q5)) + (0.02*dq1*sin(q2 + q3) - 0.4*dq2*sin(q3))*sin(q5) + (dq1*(0.4*sin(q2) + 0.45*sin(q2 + q3))*sin(q4) + (-0.02*dq1*cos(q2 + q3) + 0.4*dq2*cos(q3) + 0.45*dq2 + 0.45*dq3)*cos(q4))*cos(q5)], [x5*(dq1*sin(q4)*sin(q2 + q3) + dq5 + (dq2 + dq3)*cos(q4)) + z5*((dq1*cos(q2 + q3) + dq4)*sin(q5) + (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*cos(q5)) + (0.02*dq1*sin(q2 + q3) - 0.4*dq2*sin(q3))*cos(q5) - (dq1*(0.4*sin(q2) + 0.45*sin(q2 + q3))*sin(q4) + (-0.02*dq1*cos(q2 + q3) + 0.4*dq2*cos(q3) + 0.45*dq2 + 0.45*dq3)*cos(q4))*sin(q5)], [dq1*(0.4*sin(q2) + 0.45*sin(q2 + q3))*cos(q4) + x5*((dq1*cos(q2 + q3) + dq4)*cos(q5) - (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*sin(q5)) - y5*((dq1*cos(q2 + q3) + dq4)*sin(q5) + (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*cos(q5)) - (-0.02*dq1*cos(q2 + q3) + 0.4*dq2*cos(q3) + 0.45*dq2 + 0.45*dq3)*sin(q4)]])
    w66 = Matrix([[-((dq1*cos(q2 + q3) + dq4)*sin(q5) + (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*cos(q5))*cos(q6) + (dq1*sin(q4)*sin(q2 + q3) + dq5 + (dq2 + dq3)*cos(q4))*sin(q6)], [((dq1*cos(q2 + q3) + dq4)*sin(q5) + (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*cos(q5))*sin(q6) + (dq1*sin(q4)*sin(q2 + q3) + dq5 + (dq2 + dq3)*cos(q4))*cos(q6)], [dq6 + (dq1*cos(q2 + q3) + dq4)*cos(q5) - (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*sin(q5)]])
    vc66 = Matrix([[-y6*(dq6 + (dq1*cos(q2 + q3) + dq4)*cos(q5) - (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*sin(q5)) + z6*(((dq1*cos(q2 + q3) + dq4)*sin(q5) + (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*cos(q5))*sin(q6) + (dq1*sin(q4)*sin(q2 + q3) + dq5 + (dq2 + dq3)*cos(q4))*cos(q6)) + (dq1*(0.4*sin(q2) + 0.45*sin(q2 + q3))*cos(q4) + 0.07*(dq1*cos(q2 + q3) + dq4)*sin(q5) + 0.07*(dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*cos(q5) - (-0.02*dq1*cos(q2 + q3) + 0.4*dq2*cos(q3) + 0.45*dq2 + 0.45*dq3)*sin(q4))*sin(q6) + (0.07*dq1*sin(q4)*sin(q2 + q3) + 0.07*dq5 + 0.07*(dq2 + dq3)*cos(q4) + (0.02*dq1*sin(q2 + q3) - 0.4*dq2*sin(q3))*sin(q5) + (dq1*(0.4*sin(q2) + 0.45*sin(q2 + q3))*sin(q4) + (-0.02*dq1*cos(q2 + q3) + 0.4*dq2*cos(q3) + 0.45*dq2 + 0.45*dq3)*cos(q4))*cos(q5))*cos(q6)], [x6*(dq6 + (dq1*cos(q2 + q3) + dq4)*cos(q5) - (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*sin(q5)) + z6*(((dq1*cos(q2 + q3) + dq4)*sin(q5) + (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*cos(q5))*cos(q6) - (dq1*sin(q4)*sin(q2 + q3) + dq5 + (dq2 + dq3)*cos(q4))*sin(q6)) + (dq1*(0.4*sin(q2) + 0.45*sin(q2 + q3))*cos(q4) + 0.07*(dq1*cos(q2 + q3) + dq4)*sin(q5) + 0.07*(dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*cos(q5) - (-0.02*dq1*cos(q2 + q3) + 0.4*dq2*cos(q3) + 0.45*dq2 + 0.45*dq3)*sin(q4))*cos(q6) - (0.07*dq1*sin(q4)*sin(q2 + q3) + 0.07*dq5 + 0.07*(dq2 + dq3)*cos(q4) + (0.02*dq1*sin(q2 + q3) - 0.4*dq2*sin(q3))*sin(q5) + (dq1*(0.4*sin(q2) + 0.45*sin(q2 + q3))*sin(q4) + (-0.02*dq1*cos(q2 + q3) + 0.4*dq2*cos(q3) + 0.45*dq2 + 0.45*dq3)*cos(q4))*cos(q5))*sin(q6)], [-x6*(((dq1*cos(q2 + q3) + dq4)*sin(q5) + (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*cos(q5))*sin(q6) + (dq1*sin(q4)*sin(q2 + q3) + dq5 + (dq2 + dq3)*cos(q4))*cos(q6)) - y6*(((dq1*cos(q2 + q3) + dq4)*sin(q5) + (dq1*sin(q2 + q3)*cos(q4) - (dq2 + dq3)*sin(q4))*cos(q5))*cos(q6) - (dq1*sin(q4)*sin(q2 + q3) + dq5 + (dq2 + dq3)*cos(q4))*sin(q6)) - (0.02*dq1*sin(q2 + q3) - 0.4*dq2*sin(q3))*cos(q5) + (dq1*(0.4*sin(q2) + 0.45*sin(q2 + q3))*sin(q4) + (-0.02*dq1*cos(q2 + q3) + 0.4*dq2*cos(q3) + 0.45*dq2 + 0.45*dq3)*cos(q4))*sin(q5)]])

rc_list = [rc11, rc22, rc33, rc44, rc55, rc66]
w_list = [w11, w22, w33, w44, w55, w66]
vc_list = [vc11, vc22, vc33, vc44, vc55, vc66]
m_list = [m1, m2, m3, m4, m5, m6]
Ic_list = [Ic1, Ic2, Ic3, Ic4, Ic5, Ic6]
q_list = [q1, q2, q3, q4, q5, q6]
dq_list = [dq1, dq2, dq3, dq4, dq5, dq6]
ddq_list = [ddq1, ddq2, ddq3, ddq4, ddq5, ddq6]

k = 0
u = 0
for i in range(6):
    # 平动动能
    T_trans = 0.5 * m_list[i] * vc_list[i].dot(vc_list[i])
    # 转动动能
    T_rot = 0.5 * (w_list[i].T @ Ic_list[i] @ w_list[i])
    # 总动能
    k += T_trans + T_rot[0,0]
    # 势能
    u += m_list[i] * rc_list[i] * g

H = sp.zeros(6, 6)
for i in range(6):
    for j in range(6):
        H[i,j] = sp.diff(sp.diff(k, dq_list[i]), dq_list[j])

equations = []
M = sp.zeros(6, 6)
Cq = sp.zeros(6, 1)
G = sp.zeros(6, 1)
for i in range(6):
    # ∂k/∂dq
    dk_ddq = sp.diff(k, dq_list[i])
    # 计算全导数：链式法则 偏导求和
    d_dt_dq = 0
    d_dt_ddq = 0
    for j in range(6):
        # 对 q 的偏导
        d_dt_dq += sp.diff(dk_ddq, q_list[j]) * dq_list[j]
        # 对 dq 的偏导
        M[i,j] = sp.diff(dk_ddq, dq_list[j])
        d_dt_ddq += M[i,j] * ddq_list[j]
    
    # ∂k/∂q ∂u/∂q
    dk_dq = sp.diff(k, q_list[i])
    du_dq = sp.diff(u, q_list[i])
    G[i,0] = du_dq
    Cq[i,0] = d_dt_dq - dk_dq

    # 拉格朗日方程
    eq = d_dt_ddq + d_dt_dq - dk_dq + du_dq
    equations.append((eq))
    
print("M == H ?", simplify(M - H) == sp.zeros(6, 6))
if False:
    with open("F:\\1zhouge\\py\\robot\\sym\\sym_result_Lagrange.txt", "w") as f:
        for i in range(6):
            f.write(f"tau{i+1} = {equations[i]}\n")  # 注意要加 \n 换行

if False:
    with open("F:\\1zhouge\\py\\robot\\sym\\sym_result_Lagrange2.txt", "w") as f:
        for i in range(6):
            for j in range(6):
                f.write(f"M[{i},{j}] = {M[i, j]}\n")
        for i in range(6):
            f.write(f"Cq[{i},{0}] = {Cq[i, 0]}\n")
        for i in range(6):
            f.write(f"G[{i},{0}] = {G[i, 0]}\n")