import numpy as np
from numpy import pi
import dynamics_mdh
import dynamics_mdh2
import dynamics_min
import dynamics_min2
import dynamics_sdh
# 关闭科学计数法，保留 10 位小数
np.set_printoptions(suppress=True, precision=10, linewidth=200)
q = np.random.uniform(-1.5, 1.5, 6)
dq = np.random.uniform(-1, 1, 6)
ddq = np.random.uniform(-1, 1, 6)
m = np.array([0.31, 0.31, 0.21, 0.11, 0.11, 0.11])
x = np.array([0, 0.2, 0.2, 0, 0.03, 0])
y = np.array([0, 0.1, 0, 0.1, 0, 0])
z = np.array([0.1, 0, 0, 0.1, 0, 0.03])
I1 = np.array([0.91, 0.91, 0.91, 0, 0, 0.2])
I2 = np.array([0.91, 0.91, 0.91, 0, 0, 0.2])
I3 = np.array([0.61, 0.61, 0.61, 0, 0.2, 0])
I4 = np.array([0.61, 0.61, 0.61, 0.2, 0, 0])
I5 = np.array([0.31, 0.31, 0.31, 0.1, 0.1, 0.1])
I6 = np.array([0.31, 0.31, 0.31, 0, 0, 0])


tau1 = dynamics_mdh.calcForce(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
tau2 = dynamics_mdh2.calcForce(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
tau3 = dynamics_min.calcForce(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
tau4 = dynamics_min2.calcForce(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
tau5 = dynamics_sdh.calcForce(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)

print("1力矩传递mdh:\n", tau1)
print("2力矩传递mdh:\n", tau2)
print("3最小参数集:\n", tau3)
print("4最小参数集:\n", tau4)
print("5力矩传递sdh:\n", tau5)