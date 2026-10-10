import numpy as np
import kinematics
import dynamics_mdh
from scipy.linalg import block_diag


# 关闭科学计数法，保留 10 位小数
np.set_printoptions(suppress=True, precision=10, linewidth=200)
q = np.random.uniform(-1.5, 1.5, 6)
dq = np.random.uniform(-1, 1, 6)
ddq = np.random.uniform(-1, 1, 6)
R = kinematics.ForwardKinematics(q)[0:3, 0:3]
RR = block_diag(R, R)
w, v, dw, dv = dynamics_mdh.calcAcceleration(q, dq, ddq)

JLocal = kinematics.JacobiLocal(q)
JWorld = kinematics.JacobiWorld(q)
# 速度雅可比
vwLocal2 = JLocal @ dq.reshape(6,1)
vwWorld = JWorld @ dq.reshape(6,1)
vwLocal = RR.T @ vwWorld

# 加速度雅可比
if False:
    t = 0.001
    J1 = kinematics.JacobiWorld(q+dq*t/2)
    J2 = kinematics.JacobiWorld(q-dq*t/2)
    dJWorld = (J1-J2)/t
else:
    dJWorld = kinematics.dJacobiWorld(q, dq)
dvwWorld = JWorld @ ddq.reshape(6,1) + dJWorld @ dq.reshape(6,1)
dvwLocal = RR.T @ dvwWorld

print("传递速度:\n", v, "\n", w)
print("雅可比速度:\n", vwLocal)
print("传递加速度:\n", dv, "\n", dw)
print("雅可比加速度:\n", dvwLocal)