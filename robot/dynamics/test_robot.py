from math import sin, cos, pi
import numpy as np
import dynamics_analysis
import dynamics
import time

def MDHTrans(alpha, a, d, theta):
    """返回改进DH参数的齐次变换矩阵"""
    T = []
    for i in range(0, len(alpha)):
        T.append(np.array([
            [cos(theta[i]), -sin(theta[i]), 0, a[i]],
            [sin(theta[i])*cos(alpha[i]), cos(theta[i])*cos(alpha[i]), -sin(alpha[i]), -d[i]*sin(alpha[i])],
            [sin(theta[i])*sin(alpha[i]), cos(theta[i])*sin(alpha[i]), cos(alpha[i]), d[i]*cos(alpha[i])],
            [0, 0, 0, 1]
        ]))
    return T


def rpy_to_rot(xyz, rpy):
    T_list = []
    for p, r in zip(xyz, rpy):
        rx, ry, rz = r
        cx, sx = cos(rx), sin(rx)
        cy, sy = cos(ry), sin(ry)
        cz, sz = cos(rz), sin(rz)

        # Rz @ Ry @ Rx
        R = np.array([
            [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
            [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
            [-sy,   sx*cy,            cx*cy]
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        T_list.append(T)

    return T_list


if False:
    q = np.random.uniform(-1.5, 1.5, 6)
    dq = np.random.uniform(-1, 1, 6)
    ddq = np.random.uniform(-1, 1, 6)
else:
    q = np.array([1, 1, 1, 1, 1, 1])
    dq = np.array([0.5, 0.8, 1.1, 0.4, 0.2, 0.5])
    ddq = np.array([0, 0.2, 0, 0.5, 0, 0])

if False:
    m = np.random.uniform(0.2, 1, 6)
    x = np.random.uniform(-0.2, 0.2, 6)
    y = np.random.uniform(-0.2, 0.2, 6)
    z = np.random.uniform(-0.2, 0.2, 6)
    Ixx = np.random.uniform(0.1, 0.5, 6)
    Iyy = np.random.uniform(0.1, 0.5, 6)
    Izz = np.random.uniform(0.1, 0.5, 6)
    Ixy = np.random.uniform(0.1, 0.5, 6)
    Ixz = np.random.uniform(0.1, 0.5, 6)
    Iyz = np.random.uniform(0.1, 0.5, 6)
else:
    m = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    x = np.array([0.0, 0.2, 0.2, 0.0, 0.03, 0.0])
    y = np.array([0.0, 0.1, 0.1, 0.1, 0.0, 0.0])
    z = np.array([0.1, 0.1, 0.0, 0.1, 0.0, 0.03])
    Ixx = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    Iyy = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    Izz = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    Ixy = np.array([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01])
    Ixz = np.array([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01])
    Iyz = np.array([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01])


if False:
    # 基于mdh建立机械臂
    alpha = np.array([0, -pi/2, 0, pi/2, -pi/2, pi/2]) # 连杆扭转角
    a = np.array([0, 0, 0.4, 0, 0, 0]) # 连杆长度
    d = np.array([0, 0, 0.02, 0.45, 0, 0.07]) # 连杆偏距
    theta = np.array([0, -pi/2, pi/2, 0, 0, 0]) # 关节角
    T_list = MDHTrans(alpha, a, d, theta)
else:
    xyz = []
    rpy = []
    xyz.append(np.array([0.0, -0.0, 0.0]))
    rpy.append(np.array([0.0, -0.0, 0.0]))
    xyz.append(np.array([0.0, 0.0, 0.0]))
    rpy.append(np.array([-1.5707963267948966, -1.5707963267948966, 0]))
    xyz.append(np.array([0.4, -0.0, 0.02]))
    rpy.append(np.array([0.0, -0.0, 1.5707963267948966]))
    xyz.append(np.array([0.0, -0.45, 2.7554552980815448e-17]))
    rpy.append(np.array([1.5707963267948966, -0.0, 0.0]))
    xyz.append(np.array([0.0, 0.0, 0.0]))
    rpy.append(np.array([-1.5707963267948966, 0.0, 0.0]))
    xyz.append(np.array([0.0, -0.07, 4.286263797015737e-18]))
    rpy.append(np.array([1.5707963267948966, -0.0, 0.0]))
    T_list = rpy_to_rot(xyz, rpy)
 

robot = dynamics.Robot(T_list, x, y, z, m, Ixx, Iyy, Izz, Ixy, Ixz, Iyz)

time0 = time.perf_counter()
w, v, dw, dv = robot.calc_kinematics(q, dq, ddq, 9.81)
time1 = time.perf_counter()
v2, dv2 = robot.calc_kinematics2(q, dq, ddq, 9.81)
time2 = time.perf_counter()

print("时间1:", time1 - time0, "速度传递:\n", w, "\n", v)
print("时间2:", time2 - time1, "速度旋量传递:\n", v2[5])
print("时间1:", time1 - time0, "加速度传递:\n", dw, "\n", dv)
print("时间2:", time2 - time1, "加速度旋量传递:\n", dv2[5])

time0 = time.perf_counter()
tau1 = robot.calc_inverse_dynamics(q, dq, ddq, 9.81)
time1 = time.perf_counter()
tau2 = robot.calc_inverse_dynamics2(q, dq, ddq, 9.81)
time2 = time.perf_counter()

print("时间1:", time1 - time0, "力矩传递:\n", tau1)
print("时间2:", time2 - time1, "力矩旋量传递:\n", tau2)

time0 = time.perf_counter()
M1 = dynamics_analysis.calc_quality_matrix_MDH(q, x, y, z, m, Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
time1 = time.perf_counter()
M2 = robot.calc_M(q)
time2 = time.perf_counter()
M3 = robot.calc_M2(q)
time3 = time.perf_counter()
M4 = robot.calc_M3(q)
time4 = time.perf_counter()

print("时间1:", time1 - time0, "解析法性矩阵:\n", M1)
print("时间2:", time2 - time1, "单位向量惯惯性矩阵:\n", M2)
print("时间3:", time3 - time2, "复合刚体惯性矩阵:\n", M3)
print("时间4:", time4 - time3, "复合刚体惯性矩阵:\n", M4)

time0 = time.perf_counter()
M = robot.calc_M3(q)
c = robot.calc_C(q, dq, 9.81)
a = np.linalg.inv(M) @ (tau2 - c)
time1 = time.perf_counter()
a2 = robot.calc_forword_dynamics2(q, dq, tau2, 9.81)
time2 = time.perf_counter()


print("时间1:", time1 - time0, "复合刚体算法:\n", a)
print("时间2:", time2 - time1, "铰接体算法:\n", a2)


time0 = time.perf_counter()
k1 = robot.calc_Kinetic_nergy(q, dq)
time1 = time.perf_counter()
k2 = 0.5 * (dq @ M2 @ dq.T)
time2 = time.perf_counter()

print("时间1:", time1 - time0, "动能:\n", k1)
print("时间2:", time2 - time1, "动能旋量:\n", k2)