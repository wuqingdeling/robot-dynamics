import pinocchio as pin
import numpy as np
import kinematics
import dynamics_mdh2 as dynamics
import dynamics_analysis
from scipy.linalg import block_diag

# 关闭科学计数法，保留 10 位小数
np.set_printoptions(suppress=True, precision=10, linewidth=200)
if False:
    q = np.random.uniform(-1.5, 1.5, 6)
    dq = np.random.uniform(-1, 1, 6)
    ddq = np.random.uniform(-1, 1, 6)
else:
    q = np.array([1, 1, 1, 1, 1, 1])
    dq = np.array([0.5, 0.8, 1.1, 0.4, 0.2, 0.5])
    ddq = np.array([0, 0.2, 0, 0.5, 0, 0])

T06 = kinematics.ForwardKinematics(q)
w, v, dw, dv = dynamics.calcAcceleration(q, dq, ddq)
R = T06[0:3, 0:3]
RR = block_diag(R, R)
JWorld = kinematics.JacobiWorld(q)
JLocal = kinematics.JacobiLocal(q)

# 从 URDF 加载机器人
model = pin.buildModelFromUrdf("sym\\six_axis_arm2.urdf")
data = model.createData()
pin.forwardKinematics(model, data, q, dq, ddq)
frame_id = model.getFrameId("6_Link")
# 雅可比矩阵
# JLocal2 = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
# JWorld2 = RR @ JLocal2
# print("物体雅可比矩阵:\n", JLocal)
# print("物体雅可比矩阵2:\n", JLocal2)
# print("世界雅可比矩阵:\n", JWorld)
# print("世界雅可比矩阵2:\n", JWorld2)

RT06 = pin.updateFramePlacement(model, data, frame_id)
print("传递位置:", T06)
print("pin位置:\n", RT06.translation, "\n", RT06.rotation)

vLocal = pin.getFrameVelocity(model, data, frame_id, pin.ReferenceFrame.LOCAL)
print("传递速度:\n", v, "\n", w)
print("pin速度:\n", vLocal.linear, "\n", vLocal.angular)

aLocal = pin.getFrameClassicalAcceleration(model, data, frame_id, pin.ReferenceFrame.LOCAL)
print("传递加速度:\n", dv, "\n", dw)
print("pin加速度:\n", aLocal.linear, "\n", aLocal.angular)

# 逆动力学
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
if False:
    I = [I1, I2, I3, I4, I5, I6]
    for i in range(0, 6):
        idx = f"{i}"
        mass = m[i]
        com = np.array([x[i], y[i], z[i]])
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = I[i]
        inertia_tensor = np.array([[Ixx, -Ixy, -Ixz],
                                [-Ixy, Iyy, -Iyz],
                                [-Ixz, -Iyz, Izz]])
        inertia = pin.Inertia(mass, com, inertia_tensor)
        model.inertias[i+1] = inertia

tau1 = dynamics.calcForce(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
tau2 = dynamics_analysis.calcForce_Lagrange(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
tau3 = pin.rnea(model, data, q, dq, ddq)
print("传递力矩:\n", tau1)
print("Lagrange力矩:\n", tau2)
print("pin力矩:\n", tau3)