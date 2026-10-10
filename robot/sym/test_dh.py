import numpy as np
from numpy import pi
import dynamics_mdh2 as dynamics_mdh
import dynamics_sdh
import dynamics_min2 as dynamics_min

# 关闭科学计数法，保留 10 位小数
np.set_printoptions(suppress=True, precision=10, linewidth=200)
q = np.random.uniform(-1.5, 1.5, 6)
dq = np.random.uniform(-1, 1, 6)
ddq = np.random.uniform(-1, 1, 6)

w, v, dw, dv = dynamics_mdh.calcAcceleration(q, dq, ddq)
w2, v2, dw2, dv2 = dynamics_sdh.calcAcceleration(q, dq, ddq)
print("速度传递mdh:\n", v, "\n", w)
print("速度传递sdh:\n", v2, "\n", w2)
print("加速度传递mdh:\n", dv, "\n", dw)
print("加速度传递sdh:\n", dv2, "\n", dw2)

m = np.array([0.31, 0.31, 0.21, 0.11, 0.11, 0.11])
x = np.array([0, 0.2, 0.2, 0, 0.03, 0])
y = np.array([0, 0.1, 0, 0.1, 0, 0])
z = np.array([0.1, 0, 0, 0.1, 0, 0.03])
I1 = np.array([0.91, 0.91, 0.91, 0, 0, 0.3])
I2 = np.array([0.91, 0.91, 0.91, 0.3, 0, 0])
I3 = np.array([0.61, 0.61, 0.61, 0, 0.3, 0])
I4 = np.array([0.61, 0.61, 0.61, 0.2, 0.2, 0.2])
I5 = np.array([0.31, 0.31, 0.31, 0.1, 0.1, 0.1])
I6 = np.array([0.31, 0.31, 0.31, 0, 0, 0])


tau = dynamics_mdh.calcForce(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
tau2 = dynamics_sdh.calcForce(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
tau3 = dynamics_min.calcForce(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)

print("力矩传递mdh:\n", tau)
print("力矩传递sdh:\n", tau2)
print("最小参数集:\n", tau3)