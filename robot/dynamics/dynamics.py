from math import sin, cos, pi
import numpy as np
import dynamics_analysis
import tool as tool
import time

np.set_printoptions(suppress=True, precision=10, linewidth=200)

class Robot:
    def __init__(self, T, x, y, z, m, Ixx, Iyy, Izz, Ixy, Ixz, Iyz):
        self.DOF = len(T)
        self.T0_list = T
        self.axis = [np.array([[0], [0], [1]]) for _ in range(self.DOF)]
        
        # 惯性参数
        self.m = m   # 连杆扭转角
        self.Pc = [np.array([[x[i]], [y[i]], [z[i]]]) for i in range(self.DOF)]
        self.Ic = [np.array([
            [Ixx[i], Ixy[i], Ixz[i]],
            [Ixy[i], Iyy[i], Iyz[i]],
            [Ixz[i], Iyz[i], Izz[i]]
        ]) for i in range(self.DOF)]

        # 基于旋量的初始化
        P0 = [T[0:3, 3:4] for T in self.T0_list]
        R0 = [T[0:3, 0:3] for T in self.T0_list]
        self.Xlist = [tool.plux(R0[i].T,P0[i]) for i in range(self.DOF)]
        # 旋量轴
        self.S_list = [np.vstack([self.axis[i], np.zeros((3, 1))]) for i in range(self.DOF)]
        # 连杆惯性矩阵
        self.I = [tool.mcI_to_rbi(self.m[i], self.Pc[i], self.Ic[i]) for i in range(self.DOF)]



    def calc_kinematics(self, q, dq, ddq, g=0):
        # 坐标系变换矩阵
        T_list = []
        for i in range(6):
            T_list.append(self.T0_list[i] @ tool.rz(q[i]))
        T_list.append(np.eye(4))
        P = [T[0:3, 3:4] for T in T_list]
        R = [T[0:3, 0:3] for T in T_list]

        # 初始化
        w = [np.zeros((3, 1))] * 7
        v = [np.zeros((3, 1))] * 7
        dw = [np.zeros((3, 1))] * 7
        dv = [np.array([[0], [0], [g]])] * 7
        vc = [None] * 6
        dvc = [None] * 6

        # 外推
        for i, (Ri, Pi, dqi, ddqi, Pci, axi) in enumerate(zip(R, P, dq, ddq, self.Pc, self.axis)):
            Rt = Ri.T
            w[i+1] = Rt @ w[i] + dqi * axi
            v[i+1] = Rt @ (v[i] + tool.ccross(w[i], Pi))
            dw[i+1] = Rt @ dw[i] + ddqi * axi + tool.ccross(Rt @ w[i], dqi * axi)
            dv[i+1] = Rt @ (dv[i] + tool.ccross(dw[i], Pi) + tool.ccross(w[i], tool.ccross(w[i], Pi)))
            vc[i] = v[i+1] + tool.ccross(w[i+1], Pci)
            dvc[i] = dv[i+1] + tool.ccross(dw[i+1], Pci) + tool.ccross(w[i+1], tool.ccross(w[i+1], Pci))

        w, v, dw, dv = w[1:], v[1:], dw[1:], dv[1:]
        return w[5], v[5], dw[5], dv[5]



    def calc_kinematics2(self, q, dq, ddq, g=0):
        # 坐标系变换矩阵
        if False:
            T_list = []
            for i in range(6):
                T_list.append(self.T0_list[i] @ tool.rz(q[i]))
            P = [T[0:3, 3:4] for T in T_list]
            R = [T[0:3, 0:3] for T in T_list]
            Xlist = [tool.plux(R[i].T,P[i]) for i in range(self.DOF)]
        else:
            X_list = [tool.rotz(q[i]) @ self.Xlist[i] for i in range(self.DOF)]
        
        # 初始化
        v = [np.zeros((6, 1))] * 7
        a = [np.zeros((6, 1))] * 7
        a[0] = np.array([[0], [0], [0], [0], [0], [g]])

        # 外推
        for i in range(6):
            v[i+1] = X_list[i] @ v[i] + self.S_list[i] * dq[i]
            a[i+1] = X_list[i] @ a[i] + self.S_list[i] * ddq[i] + tool.crm(v[i+1]) @ (self.S_list[i] * dq[i])
        
        return v[1:], a[1:]



    def calc_inverse_dynamics(self, q, dq, ddq, g=0):
        # 坐标系变换矩阵
        T_list = []
        for i in range(6):
            T_list.append(self.T0_list[i] @ tool.rz(q[i]))
        T_list.append(np.eye(4))
        P = [T[0:3, 3:4] for T in T_list]
        R = [T[0:3, 0:3] for T in T_list]

        # 初始化
        w = [np.zeros((3, 1))] * (self.DOF + 1)
        v = [np.zeros((3, 1))] * (self.DOF + 1)
        dw = [np.zeros((3, 1))] * (self.DOF + 1)
        dv = [np.array([[0], [0], [g]])] * (self.DOF + 1)
        dvc = [None] * self.DOF

        # 外推
        for i, (Ri, Pi, dqi, ddqi, Pci, axi) in enumerate(zip(R, P, dq, ddq, self.Pc, self.axis)):
            Rt = Ri.T
            w[i+1] = Rt @ w[i] + dqi * axi
            v[i+1] = Rt @ (v[i] + tool.ccross(w[i], Pi))
            dw[i+1] = Rt @ dw[i] + tool.ccross(Rt @ w[i], dqi * axi) + ddqi * axi
            dv[i+1] = Rt @ (dv[i] + tool.ccross(dw[i], Pi) + tool.ccross(w[i], tool.ccross(w[i], Pi)))
            dvc[i] = dv[i+1] + tool.ccross(dw[i+1], Pci) + tool.ccross(w[i+1], tool.ccross(w[i+1], Pci))
        w, v, dw, dv = w[1:], v[1:], dw[1:], dv[1:]

        # 初始化
        Fc = [None] * self.DOF
        Nc = [None] * self.DOF

        # 外推
        for i, (wi, dwi, dvci, mi, Ici) in enumerate(zip(w, dw, dvc, self.m, self.Ic)):
            Fc[i] = mi * dvci
            Nc[i] = Ici @ dwi + tool.ccross(wi, Ici @ wi)
            # print(np.vstack([Nc[i], Fc[i]]))

        # 初始化
        f = [None] * (self.DOF + 1)
        n = [None] * (self.DOF + 1)
        f[6] = np.array([[0], [0], [0]])
        n[6] = np.array([[0], [0], [0]])

        # 内推
        for i in range(self.DOF - 1, -1, -1):
            f[i] = R[i+1] @ f[i+1] + Fc[i]
            n[i] = R[i+1] @ n[i+1] + tool.ccross(P[i+1], R[i+1] @ f[i+1]) + Nc[i] + tool.ccross(self.Pc[i], Fc[i])

        # 关节力矩
        tau = np.zeros(self.DOF)
        for i in range(self.DOF):
            tau[i] = (self.axis[i].T @ n[i]).item()
        return tau



    def calc_inverse_dynamics2(self, q, dq, ddq, g=0):
        # 坐标系变换矩阵
        if False:
            T_list = []
            for i in range(6):
                T_list.append(self.T0_list[i] @ tool.rz(q[i]))
            P = [T[0:3, 3:4] for T in T_list]
            R = [T[0:3, 0:3] for T in T_list]
            X_list = [tool.plux(R[i].T,P[i]) for i in range(self.DOF)]
        else:
            X_list = [tool.rotz(q[i]) @ self.Xlist[i] for i in range(self.DOF)]
        
        # 初始化
        v = [np.zeros((6, 1))] * (self.DOF + 1)
        a = [np.zeros((6, 1))] * (self.DOF + 1)
        f = [np.zeros((6, 1))] * self.DOF
        a[0] = np.array([[0], [0], [0], [0], [0], [g]])

        # 外推
        for i in range(self.DOF):
            v[i+1] = X_list[i] @ v[i] + self.S_list[i] * dq[i]
            a[i+1] = X_list[i] @ a[i] + self.S_list[i] * ddq[i] + tool.crm(v[i+1]) @ (self.S_list[i] * dq[i])
            f[i] = self.I[i] @ a[i+1] + tool.crf(v[i+1]) @ self.I[i] @ v[i+1]

        # 内推
        for i in range(self.DOF - 1, 0, -1):
            f[i-1] = f[i-1] + X_list[i].T @ f[i]

        # 关节力矩
        tau = np.zeros(self.DOF)
        for i in range(self.DOF):
            tau[i] = (self.S_list[i].T @ f[i]).item()
        
        return tau



    def calc_M(self, q):
        M = np.zeros((6, 6))
        dq = np.zeros(6)
        for i in range(0, 6):
            ddq = np.eye(6)[i]
            tau = self.calc_inverse_dynamics(q, dq, ddq)
            M[:,i] = tau.flatten()
        return M



    def calc_M2(self, q):
        # 坐标系变换矩阵
        T_list = []
        for i in range(6):
            T_list.append(self.T0_list[i] @ tool.rz(q[i]))
        T_list.append(np.eye(4))
        P = [T[0:3, 3:4] for T in T_list]
        R = [T[0:3, 0:3] for T in T_list]
        X_list = [tool.plux(R[i].T,P[i]) for i in range(self.DOF)]

        m_new = self.m.copy() 
        Pc_new = self.Pc.copy() 
        Ic_new = self.Ic.copy()

        for i in reversed(range(1, len(R)-1)):     # i = 5,4,3,2,1
            Ri, Pi = R[i], P[i]
            mi, Pci, Ici = self.m[i-1], self.Pc[i-1], self.Ic[i-1]
            m_nexti, Pc_nexti, Ic_nexti = m_new[i], Pc_new[i], Ic_new[i]
            m_new[i-1]  = mi + m_nexti
            Pc_new[i-1] = (m_nexti * (Pi + Ri @ Pc_nexti) + mi * Pci) / (m_nexti + mi)
            P1 = Pi + Ri @ Pc_nexti - Pc_new[i-1]
            I_new_1 = Ri @ Ic_nexti @ Ri.T + m_nexti * ((P1.T @ P1) * np.eye(3) - P1 @ P1.T)
            P2 = Pci - Pc_new[i-1]
            I_new_2 = Ici + mi * ((P2.T @ P2) * np.eye(3) - P2 @ P2.T)
            Ic_new[i-1] = I_new_1 + I_new_2

        I6_new = []
        for i in range(6):
            G = np.zeros((6, 6))
            G[:3, :3] = Ic_new[i]
            G[3:, 3:] = m_new[i] * np.eye(3)
            G = tool.mcI_to_rbi(m_new[i], Pc_new[i], Ic_new[i])
            I6_new.append(G)
        
        H = np.zeros((self.DOF, self.DOF))
        
        for i in range(self.DOF):
            fh = I6_new[i] @ self.S_list[i]
            H[i, i] = (self.S_list[i].T @ fh).item()
            for j in range(i, 0, -1):
                fh = X_list[j].T @ fh
                j = j-1
                H[i, j] = (self.S_list[j].T @ fh).item()
                H[j, i] = H[i, j]

        return H



    def calc_M3(self, q):
        # 坐标系变换矩阵
        if False:
            T_list = []
            for i in range(6):
                T_list.append(self.T0_list[i] @ tool.rz(q[i]))
            P = [T[0:3, 3:4] for T in T_list]
            R = [T[0:3, 0:3] for T in T_list]
            X_list = [tool.plux(R[i].T,P[i]) for i in range(self.DOF)]
        else:
            X_list = [tool.rotz(q[i]) @ self.Xlist[i] for i in range(self.DOF)]

        IA = self.I.copy()
        for i in range(self.DOF - 1, 0, -1):
            IA[i-1] = IA[i-1] + X_list[i].T @ IA[i] @ X_list[i]

        H = np.zeros((self.DOF, self.DOF))

        for i in range(self.DOF):
            fh = IA[i] @ self.S_list[i]
            H[i, i] = (self.S_list[i].T @ fh).item()
            for j in range(i, 0, -1):
                fh = X_list[j].T @ fh
                j = j-1
                H[i, j] = (self.S_list[j].T @ fh).item()
                H[j, i] = H[i, j]

        return H
    

    def calc_C(self, q, dq, g):
        # 坐标系变换矩阵
        if False:
            T_list = []
            for i in range(6):
                T_list.append(self.T0_list[i] @ tool.rz(q[i]))
            P = [T[0:3, 3:4] for T in T_list]
            R = [T[0:3, 0:3] for T in T_list]
            X_list = [tool.plux(R[i].T,P[i]) for i in range(self.DOF)]
        else:
            X_list = [tool.rotz(q[i]) @ self.Xlist[i] for i in range(self.DOF)]

        # 初始化
        v = [np.zeros((6, 1))] * (self.DOF + 1)
        a = [np.zeros((6, 1))] * (self.DOF + 1)
        f = [np.zeros((6, 1))] * self.DOF
        a[0] = np.array([[0], [0], [0], [0], [0], [g]])

        # 外推
        for i in range(self.DOF):
            v[i+1] = X_list[i] @ v[i] + self.S_list[i] * dq[i]
            a[i+1] = X_list[i] @ a[i] + tool.crm(v[i+1]) @ (self.S_list[i] * dq[i])
            f[i] = self.I[i] @ a[i+1] + tool.crf(v[i+1]) @ self.I[i] @ v[i+1]

        # 内推
        for i in range(self.DOF - 1, 0, -1):
            f[i-1] = f[i-1] + X_list[i].T @ f[i]

        # 关节力矩
        C = np.zeros(self.DOF)
        for i in range(self.DOF):
            C[i] = (self.S_list[i].T @ f[i]).item()
        
        return C


    def calc_forword_dynamics2(self, q, dq, tau, g=0):
        # 坐标系变换矩阵
        if False:
            T_list = []
            for i in range(6):
                T_list.append(self.T0_list[i] @ tool.rz(q[i]))
            P = [T[0:3, 3:4] for T in T_list]
            R = [T[0:3, 0:3] for T in T_list]
            X_list = [tool.plux(R[i].T,P[i]) for i in range(self.DOF)]
        else:
            X_list = [tool.rotz(q[i]) @ self.Xlist[i] for i in range(self.DOF)]

        # 初始化
        v = [np.zeros((6, 1))] * self.DOF
        c = [np.zeros((6, 1))] * self.DOF
        pA = [np.zeros((6, 1))] * self.DOF

        # 外推速度 偏置加速度
        for i in range(self.DOF):
            if i==0:
                v[i] = self.S_list[i] * dq[i]
                c[i] = np.zeros((6, 1))
            else:
                v[i] = X_list[i] @ v[i-1] + self.S_list[i] * dq[i]
                c[i] = tool.crm(v[i]) @ (self.S_list[i] * dq[i])
            pA[i] = tool.crf(v[i]) @ self.I[i] @ v[i]

        IA = self.I.copy()
        U = [None] * self.DOF
        d = np.zeros(self.DOF)
        u = np.zeros(self.DOF)

        # 内推 偏置力
        for i in range(self.DOF - 1, -1, -1):
            U[i] = IA[i] @ self.S_list[i]
            d[i] = (self.S_list[i].T @ U[i]).item()
            u[i] = tau[i] - (self.S_list[i].T @ pA[i]).item()
            if i != 0:
                p = i-1
                Ia = IA[i] - U[i]/d[i] @ U[i].T
                pa = pA[i] + Ia @ c[i] + U[i] * u[i]/d[i]
                IA[p] = IA[p] + X_list[i].T @ Ia @ X_list[i]
                pA[p] = pA[p] + X_list[i].T @ pa

        # 外推 关节加速度
        qdd = np.zeros(self.DOF)
        a = [None] * self.DOF
        for i in range(self.DOF):
            if i == 0:
                a[i] = X_list[i] @ np.array([[0], [0], [0], [0], [0], [g]]) + c[i]
            else:
                a[i] = X_list[i] @ a[i-1] + c[i]
            qdd[i] = (u[i] - (U[i].T @ a[i]).item()) / d[i]
            a[i] = a[i] + self.S_list[i] * qdd[i]

        return qdd


    def calc_Kinetic_nergy(self, q, dq):
        # 坐标系变换矩阵
        T_list = []
        for i in range(6):
            T_list.append(self.T0_list[i] @ tool.rz(q[i]))
        T_list.append(np.eye(4))
        P = [T[0:3, 3:4] for T in T_list]
        R = [T[0:3, 0:3] for T in T_list]

        w = [np.zeros((3, 1))] * 7
        v = [np.zeros((3, 1))] * 7
        vc = [None] * 6
        for i, (Ri, Pi, dqi, Pci, axi) in enumerate(zip(R, P, dq, self.Pc, self.axis)):
            Rt = Ri.T
            w[i+1] = Rt @ w[i] + dqi * axi
            v[i+1] = Rt @ (v[i] + tool.ccross(w[i], Pi))
            vc[i] = v[i+1] + tool.ccross(w[i+1], Pci)
        w, v = w[1:], v[1:]
        k = 0
        for i in range(6):
            # 平动动能
            T_trans = 0.5 * self.m[i] * (vc[i].T @ vc[i]).item()
            # 转动动能
            T_rot = 0.5 * (w[i].T @ self.Ic[i] @ w[i])
            # 总动能
            k += T_trans + T_rot.item()
        return k


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


if __name__ == "__main__":
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


    alpha = np.array([0, -pi/2, 0, pi/2, -pi/2, pi/2]) # 连杆扭转角
    a = np.array([0, 0, 0.4, 0, 0, 0]) # 连杆长度
    d = np.array([0, 0, 0.02, 0.45, 0, 0.07]) # 连杆偏距
    theta = np.array([0, -pi/2, pi/2, 0, 0, 0]) # 关节角

    T_list = MDHTrans(alpha, a, d, theta)

    robot = Robot(T_list, x, y, z, m, Ixx, Iyy, Izz, Ixy, Ixz, Iyz)

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