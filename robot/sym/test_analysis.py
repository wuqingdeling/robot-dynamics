import numpy as np
from numpy import pi
import time
import dynamics_mdh2 as dynamics_mdh
import dynamics_min2 as dynamics_min
import dynamics_analysis

# 关闭科学计数法，保留 10 位小数
np.set_printoptions(suppress=True, precision=10, linewidth=200)
if False:
    q = np.array([0, 0, 0, 0, 0, 0])
    dq = np.array([0, 0, 0, 0, 0, 0])
    ddq = np.array([1, 1, 1, 1, 1, 1])
else:
    q = np.random.uniform(-1.5, 1.5, 6)
    dq = np.random.uniform(-1, 1, 6)
    ddq = np.array([0, 0, 0, 0, 0, 0])

w, v, dw, dv = dynamics_mdh.calcAcceleration(q, dq, ddq)
w2, v2, dw2, dv2 = dynamics_analysis.calcAcceleration(q, dq, ddq)
print("速度传递:\n", v, "\n", w)
print("速度解析:\n", v2, "\n", w2)
print("加速度传递:\n", dv, "\n", dw)
print("加速度解析:\n", dv2, "\n", dw2)

if True:
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
    m = np.array([0.31, 0.31, 0.21, 0.11, 0.11, 0.11])
    x = np.array([0, 0.2, 0.2, 0, 0.03, 0])
    y = np.array([0, 0.1, 0.1, 0.1, 0, 0])
    z = np.array([0, 0.1, 0, 0.1, 0, 0.03])
    I1 = np.array([0.61, 0.62, 0.63, 0, 0, 0])
    I2 = np.array([0.51, 0.52, 0.53, 0, 0, 0])
    I3 = np.array([0.41, 0.42, 0.43, 0, 0, 0])
    I4 = np.array([0.31, 0.32, 0.33, 0, 0, 0])
    I5 = np.array([0.21, 0.22, 0.23, 0, 0, 0])
    I6 = np.array([0.11, 0.12, 0.13, 0, 0, 0])


start0 = time.perf_counter()
tau1 = dynamics_mdh.calcForce(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
start1 = time.perf_counter()
tau2 = dynamics_analysis.calcForce(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
start2 = time.perf_counter()
tau3 = dynamics_min.calcForce(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
start3 = time.perf_counter()
tau4 = dynamics_analysis.calcForce_min(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
start4 = time.perf_counter()
tau5 = dynamics_analysis.calcForce_Lagrange(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
start5 = time.perf_counter()
tau6 = dynamics_analysis.calcForce_Lagrange2(q, dq, ddq, 9.81, x, y, z, m, I1, I2, I3, I4, I5, I6)
start6 = time.perf_counter()

print("时间1:", start1 - start0, "力矩传递:\n", tau1)
print("时间2:", start2 - start1, "力矩解析:\n", tau2)
print("时间3:", start3 - start2, "力矩传递min:\n", tau3)
print("时间4:", start4 - start3, "力矩解析min:\n", tau4)
print("时间5:", start5 - start4, "力矩Lagrange:\n", tau5)
print("时间6:", start6 - start5, "力矩Lagrange2:\n", tau6)