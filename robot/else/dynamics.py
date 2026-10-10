from math import sin, cos, pi
import numpy as np
import tool
import time

np.set_printoptions(suppress=True, precision=10, linewidth=200)

class SixAxisRobot:
    def __init__(self, x=(0,0,0,0,0,0), y=(0,0,0,0,0,0), z=(0,0,0,0,0,0), m=(0,0,0,0,0,0), I1=(0,0,0,0,0,0), I2=(0,0,0,0,0,0), I3=(0,0,0,0,0,0), I4=(0,0,0,0,0,0), I5=(0,0,0,0,0,0), I6=(0,0,0,0,0,0)):
        DOF = 6
        self.DOF = DOF
        # 改进 DH 参数
        self.alpha = np.array([0, -pi/2, 0, pi/2, -pi/2, pi/2]) # 连杆扭转角
        self.a = np.array([0, 0, 0.4, 0, 0, 0]) # 连杆长度
        self.d = np.array([0, 0, 0.02, 0.45, 0, 0.07]) # 连杆偏距
        self.theta = np.array([0, -pi/2, pi/2, 0, 0, 0]) # 关节角
        self.axis = [np.array([[0], [0], [1]]) for _ in range(6)]
        
        # 惯性参数
        self.m = m   # 连杆扭转角
        self.Pc = [np.array([[x[i]], [y[i]], [z[i]]]) for i in range(6)]
        I_List = [I1, I2, I3, I4, I5, I6]
        self.Ic = [np.array([
            [I_List[i][0], -I_List[i][3], -I_List[i][4]],
            [-I_List[i][3], I_List[i][1], -I_List[i][5]],
            [-I_List[i][4], -I_List[i][5], I_List[i][2]]
        ]) for i in range(6)]

        # 基于旋量的初始化
        T_position = self.calc_position(np.zeros(6))
        # ============================================================
        # T_C[DOF]: 连杆质心坐标
        # inertial_G[DOF]: 惯性矩阵
        # ============================================================
        self.T_C = []
        for i in range(DOF):
            T = np.eye(4)
            T[:3, 3:4] = self.Pc[i]
            self.T_C.append(T)
        self.Trans_Cass = np.asarray(m).reshape(DOF)
        self.inertial_G = []
        for i in range(DOF):
            G = np.zeros((6, 6))
            G[:3, :3] = self.Ic[i]
            G[3:, 3:] = self.Trans_Cass[i] * np.eye(3)
            self.inertial_G.append(G)
        # ============================================================
        # links_S[DOF]: 连杆旋量
        # links_B[DOF]: 基座旋量
        # links_C[DOF]: 质心旋量
        # ============================================================
        self.links_S = [np.vstack([axi, np.zeros((3, 1))]) for axi in self.axis]
        self.links_B = []
        self.links_C = []
        for i in range(DOF):
            B = tool.AD_SE3(T_position[i]) @ self.links_S[i]
            self.links_B.append(B)
            C = tool.AD_SE3(tool.inv_SE3(self.T_C[i])) @ self.links_S[i]
            self.links_C.append(C)
        # ============================================================
        # 6. cor_M[DOF]: 质心基座位姿
        # 7. Trans_C[DOF]: 质心相对上一质心位姿
        # ============================================================
        cor_M = []
        for i in range(DOF):
            cor_M.append(T_position[i] @ self.T_C[i])
        self.Trans_C = []
        for i in range(DOF):
            if i == 0:
                self.Trans_C.append(cor_M[0])
            else:
                self.Trans_C.append(tool.inv_SE3(cor_M[i - 1]) @ cor_M[i])

    def mdh_trans(self, q=None):
        """返回各关节的改进 DH 齐次变换矩阵列表"""
        theta = self.theta.copy() if q is None else self.theta + np.asarray(q)
        T = []
        for i in range(6):
            T.append(np.array([
                [cos(theta[i]), -sin(theta[i]), 0, self.a[i]],
                [sin(theta[i])*cos(self.alpha[i]),  cos(theta[i])*cos(self.alpha[i]), -sin(self.alpha[i]), -self.d[i]*sin(self.alpha[i])],
                [sin(theta[i])*sin(self.alpha[i]),  cos(theta[i])*sin(self.alpha[i]),  cos(self.alpha[i]),  self.d[i]*cos(self.alpha[i])],
                [0, 0, 0, 1]
            ]))
        return T

    def calc_position(self, q):
        T_list = self.mdh_trans(q)
        for i in range(1, 6):
            T_list[i] = T_list[i-1] @ T_list[i]
        return T_list

    def calc_acceleration(self, q, dq, ddq, g=0):
        # 坐标系变换矩阵
        T_list = self.mdh_trans(q)
        T_list.append(np.eye(4))
        P = [T[0:3, 3:4] for T in T_list]
        R = [T[0:3, 0:3] for T in T_list]

        w = [np.zeros((3, 1))] * 7
        v = [np.zeros((3, 1))] * 7
        dw = [np.zeros((3, 1))] * 7
        dv = [np.array([[0], [0], [g]])] * 7
        vc = [None] * 6
        dvc = [None] * 6
        for i, (Ri, Pi, dqi, ddqi, Pci, axi) in enumerate(zip(R, P, dq, ddq, self.Pc, self.axis)):
            Rt = Ri.T
            w[i+1] = Rt @ w[i] + dqi * axi
            v[i+1] = Rt @ (v[i] + tool.ccross(w[i], Pi))
            dw[i+1] = Rt @ dw[i] + ddqi * axi + tool.ccross(Rt @ w[i], dqi * axi)
            dv[i+1] = Rt @ (dv[i] + tool.ccross(dw[i], Pi) + tool.ccross(w[i], tool.ccross(w[i], Pi)))
            vc[i] = v[i+1] + tool.ccross(w[i+1], Pci)
            dvc[i] = dv[i+1] + tool.ccross(dw[i+1], Pci) + tool.ccross(w[i+1], tool.ccross(w[i+1], Pci))

        w, v, dw, dv = w[1:], v[1:], dw[1:], dv[1:]
        return w[5], vc[5], dw[5], dvc[5]


    def calc_acceleration2(self, q, dq, ddq, g=0):
        DOF = self.DOF
        # ---------- 迭代变量 ----------
        Tc = [np.eye(4) for _ in range(DOF + 1)]
        vc = [np.zeros((6, 1)) for _ in range(DOF + 1)]
        ac = [np.zeros((6, 1)) for _ in range(DOF + 1)]

        # ---------- 初始条件 ----------
        vc[0] = np.zeros((6, 1))
        ac[0] = np.array([[0], [0], [0], [0], [0], [g]])

        # ---------- 前向递推：速度、加速度 ----------
        for i in range(DOF):
            # 当前位形下，质心 i 相对质心 i-1 的变换
            Tc[i] = tool.exp_twist_2_SE3(-self.links_C[i], q[i]) @ tool.inv_SE3(self.Trans_C[i])
            # 速度递推:
            vc[i + 1] = (tool.AD_SE3(Tc[i]) @ vc[i] + self.links_C[i] * dq[i])
            # 加速度递推:
            ac[i + 1] = (tool.AD_SE3(Tc[i]) @ ac[i] + tool.ad_screw(vc[i + 1]) @ self.links_C[i] * dq[i] + self.links_C[i] * ddq[i])

        return vc[1:], ac[1:]

    def calc_acceleration3(self, q, dq, ddq, g=0):
        # 坐标系变换矩阵
        T_list = self.mdh_trans(q)
        P = [T[0:3, 3:4] for T in T_list]
        R = [T[0:3, 0:3] for T in T_list]

        X_list = [tool.build_X(R[i].T, P[i]) for i in range(6)]
        S_list = [np.vstack([axi, np.zeros((3, 1))]) for axi in self.axis]

        # 初始化空间矢量 (6x1)
        v = [np.zeros((6, 1))] * 7
        a = [np.zeros((6, 1))] * 7
        a[0] = np.array([[0], [0], [0], [0], [0], [g]])  # 重力作为基座加速度

        for i in range(6):
            # 计算当前连杆速度
            v[i+1] = X_list[i] @ v[i] + S_list[i] * dq[i]
            # 计算当前连杆加速度
            a[i+1] = X_list[i] @ a[i] + S_list[i] * ddq[i] + tool.spatial_cross(X_list[i] @ v[i], v[i+1])
            
        return v[1:], a[1:]


    def calc_tau(self, q, dq, ddq, g=0):
        # 坐标系变换矩阵
        T_list = self.mdh_trans(q)
        T_list.append(np.eye(4))
        P = [T[0:3, 3:4] for T in T_list]
        R = [T[0:3, 0:3] for T in T_list]

        w = [np.zeros((3, 1))] * 7
        v = [np.zeros((3, 1))] * 7
        dw = [np.zeros((3, 1))] * 7
        dv = [np.array([[0], [0], [g]])] * 7
        dvc = [None] * 6
        for i, (Ri, Pi, dqi, ddqi, Pci, axi) in enumerate(zip(R, P, dq, ddq, self.Pc, self.axis)):
            Rt = Ri.T
            w[i+1] = Rt @ w[i] + dqi * axi
            v[i+1] = Rt @ (v[i] + tool.ccross(w[i], Pi))
            dw[i+1] = Rt @ dw[i] + tool.ccross(Rt @ w[i], dqi * axi) + ddqi * axi
            dv[i+1] = Rt @ (dv[i] + tool.ccross(dw[i], Pi) + tool.ccross(w[i], tool.ccross(w[i], Pi)))
            dvc[i] = dv[i+1] + tool.ccross(dw[i+1], Pci) + tool.ccross(w[i+1], tool.ccross(w[i+1], Pci))
        w, v, dw, dv = w[1:], v[1:], dw[1:], dv[1:]

        Fc = [None] * 6
        Nc = [None] * 6
        for i, (wi, dwi, dvci, mi, Ici) in enumerate(zip(w, dw, dvc, self.m, self.Ic)):
            Fc[i] = mi * dvci
            Nc[i] = Ici @ dwi + tool.ccross(wi, Ici @ wi)

        f = [None] * 7
        n = [None] * 7
        f[6] = np.array([[0], [0], [0]])
        n[6] = np.array([[0], [0], [0]])
        for i in range(5, -1, -1):
            f[i] = R[i+1] @ f[i+1] + Fc[i]
            n[i] = R[i+1] @ n[i+1] + tool.ccross(P[i+1], R[i+1] @ f[i+1]) + Nc[i] + tool.ccross(self.Pc[i], Fc[i])

        tau = np.array([[n[0][2,0]], [n[1][2,0]], [n[2][2,0]], [n[3][2,0]], [n[4][2,0]], [n[5][2,0]]])
        return tau

    def calc_tau2(self, q, dq, ddq, g=0):
        # 坐标系变换矩阵
        T_list = self.mdh_trans(q)
        T_list.append(np.eye(4))
        P = [T[0:3, 3:4] for T in T_list]
        R = [T[0:3, 0:3] for T in T_list]

        # 外推：连杆角速度、角加速度、线加速度、质心加速度、质心受力
        w00 = np.array([[0], [0], [0]])
        v00 = np.array([[0], [0], [0]])
        dw00 = np.array([[0], [0], [0]])
        dv00 = np.array([[0], [0], [g]])
        w11, v11, dw11, dv11, dvc11 = tool.motion_clc(R[0].T, P[0], dq[0], ddq[0], w00, v00, dw00, dv00, self.Pc[0], self.axis[0])
        w22, v22, dw22, dv22, dvc22 = tool.motion_clc(R[1].T, P[1], dq[1], ddq[1], w11, v11, dw11, dv11, self.Pc[1], self.axis[1])
        w33, v33, dw33, dv33, dvc33 = tool.motion_clc(R[2].T, P[2], dq[2], ddq[2], w22, v22, dw22, dv22, self.Pc[2], self.axis[2])
        w44, v44, dw44, dv44, dvc44 = tool.motion_clc(R[3].T, P[3], dq[3], ddq[3], w33, v33, dw33, dv33, self.Pc[3], self.axis[3])
        w55, v55, dw55, dv55, dvc55 = tool.motion_clc(R[4].T, P[4], dq[4], ddq[4], w44, v44, dw44, dv44, self.Pc[4], self.axis[4])
        w66, v66, dw66, dv66, dvc66 = tool.motion_clc(R[5].T, P[5], dq[5], ddq[5], w55, v55, dw55, dv55, self.Pc[5], self.axis[5])

        Fc11, Nc11 = tool.FN_clc(w11, dw11, dvc11, self.m[0], self.Ic[0])
        Fc22, Nc22 = tool.FN_clc(w22, dw22, dvc22, self.m[1], self.Ic[1])
        Fc33, Nc33 = tool.FN_clc(w33, dw33, dvc33, self.m[2], self.Ic[2])
        Fc44, Nc44 = tool.FN_clc(w44, dw44, dvc44, self.m[3], self.Ic[3])
        Fc55, Nc55 = tool.FN_clc(w55, dw55, dvc55, self.m[4], self.Ic[4])
        Fc66, Nc66 = tool.FN_clc(w66, dw66, dvc66, self.m[5], self.Ic[5])

        f77 = np.array([[0], [0], [0]])
        n77 = np.array([[0], [0], [0]])
        f66, n66 = tool.fn_clc(R[6], P[6], Fc66, Nc66, f77, n77, self.Pc[5])
        f55, n55 = tool.fn_clc(R[5], P[5], Fc55, Nc55, f66, n66, self.Pc[4])
        f44, n44 = tool.fn_clc(R[4], P[4], Fc44, Nc44, f55, n55, self.Pc[3])
        f33, n33 = tool.fn_clc(R[3], P[3], Fc33, Nc33, f44, n44, self.Pc[2])
        f22, n22 = tool.fn_clc(R[2], P[2], Fc22, Nc22, f33, n33, self.Pc[1])
        f11, n11 = tool.fn_clc(R[1], P[1], Fc11, Nc11, f22, n22, self.Pc[0])

        tau = np.array([[n11[2,0]], [n22[2,0]], [n33[2,0]], [n44[2,0]], [n55[2,0]], [n66[2,0]]])
        return tau

    def calc_tau3(self, q, dq, ddq, g=0):
        DOF = self.DOF
        # ---------- 迭代变量 ----------
        Tc = [np.eye(4) for _ in range(DOF + 1)]
        vc = [np.zeros((6, 1)) for _ in range(DOF + 1)]
        ac = [np.zeros((6, 1)) for _ in range(DOF + 1)]
        vc[0] = np.zeros((6, 1))
        ac[0] = np.array([[0], [0], [0], [0], [0], [g]])

        # ---------- 前向递推：速度、加速度 ----------
        for i in range(DOF):
            # 当前位形下，质心 i 相对质心 i-1 的变换
            Tc[i] = tool.exp_twist_2_SE3(-self.links_C[i], q[i]) @ tool.inv_SE3(self.Trans_C[i])
            # 速度递推:
            vc[i + 1] = (tool.AD_SE3(Tc[i]) @ vc[i] + self.links_C[i] * dq[i])
            # 加速度递推:
            ac[i + 1] = (tool.AD_SE3(Tc[i]) @ ac[i] + tool.ad_screw(vc[i + 1]) @ self.links_C[i] * dq[i] + self.links_C[i] * ddq[i])

        # ---------- 迭代变量 ----------
        F = [np.zeros((6, 1)) for _ in range(DOF + 1)]
        tau = np.zeros(DOF)
        F[DOF] = np.zeros((6, 1))
        Tc[DOF] = np.eye(4)

        # ---------- 后向递推：力力矩 ----------
        for i in range(DOF - 1, -1, -1):
            # 力递推:
            F[i] = (tool.AD_SE3(Tc[i + 1]).T @ F[i + 1] + self.inertial_G[i] @ ac[i + 1] - tool.ad_screw(vc[i + 1]).T @ self.inertial_G[i] @ vc[i + 1])
            # 关节力矩:
            tau[i] = (F[i].T @ self.links_C[i]).item()

        return tau


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
        I1 = np.random.uniform(0.1, 0.5, 6)
        I2 = np.random.uniform(0.1, 0.5, 6)
        I3 = np.random.uniform(0.1, 0.5, 6)
        I4 = np.random.uniform(0.1, 0.5, 6)
        I5 = np.random.uniform(0.1, 0.5, 6)
        I6 = np.random.uniform(0.1, 0.5, 6)
    else:
        m = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        x = np.array([0.0, 0.2, 0.2, 0.0, 0.03, 0.0])
        y = np.array([0.0, 0.1, 0.1, 0.1, 0.0, 0.0])
        z = np.array([0.1, 0.1, 0.0, 0.1, 0.0, 0.03])
        I1 = np.array([0.6, 0.6, 0.6, 0.01, 0.01, 0.01])
        I2 = np.array([0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
        I3 = np.array([0.4, 0.4, 0.4, 0.01, 0.01, 0.01])
        I4 = np.array([0.3, 0.3, 0.3, 0.01, 0.01, 0.01])
        I5 = np.array([0.2, 0.2, 0.2, 0.01, 0.01, 0.01])
        I6 = np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

    robot = SixAxisRobot(x, y, z, m, I1, I2, I3, I4, I5, I6)

    time0 = time.perf_counter()
    w, v, dw, dv = robot.calc_acceleration(q, dq, ddq, 9.81)
    time1 = time.perf_counter()
    v2, dv2 = robot.calc_acceleration2(q, dq, ddq, 9.81)
    time2 = time.perf_counter()
    v3, dv3 = robot.calc_acceleration3(q, dq, ddq, 9.81)
    time3 = time.perf_counter()

    print("时间1:", time1 - time0, "速度传递:\n", w, "\n", v)
    print("时间2:", time2 - time1, "速度旋量传递:\n", v2[5])
    print("时间3:", time3 - time2, "速度旋量传递:\n", v3[5])
    
    print("时间1:", time1 - time0, "加速度传递:\n", dw, "\n", dv)
    print("时间2:", time2 - time1, "加速度旋量传递:\n", dv2[5])
    print("时间3:", time3 - time2, "加速度旋量传递:\n", dv3[5])
    
    time0 = time.perf_counter()
    tau1 = robot.calc_tau(q, dq, ddq, 9.81)
    time1 = time.perf_counter()
    tau2 = robot.calc_tau2(q, dq, ddq, 9.81)
    time2 = time.perf_counter()
    tau3 = robot.calc_tau3(q, dq, ddq, 9.81)
    time3 = time.perf_counter()

    print("时间1:", time1 - time0, "力矩传递:\n", tau1)
    print("时间2:", time2 - time1, "力矩传递:\n", tau2)
    print("时间3:", time3 - time2, "力矩旋量传递:\n", tau3)