import numpy as np
from math import sin, cos
from scipy.linalg import block_diag
import time

# 连杆扭转角alpha     连杆长度a       连杆偏距d       关节角theta
#       0               0               0               0
#       -90	            0	            0	            -90
#       0	            400	            20	            90
#       90	            0	            450	            0
#       -90	            0	            0	            0
#       90	            0	            70	            0

def rotx(alpha):
    T = np.array([[1, 0, 0, 0], 
                [0, cos(alpha), -sin(alpha), 0], 
                [0, sin(alpha), cos(alpha), 0],
                [0, 0, 0, 1]])
    return T

def trsfx(a):
    T = np.array([[1, 0, 0, a], 
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    return T

def trsfz(d):
    T = np.array([[1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 1, d],
                [0, 0, 0, 1]])
    return T

def rotz(theta):
    T = np.array([[cos(theta), -sin(theta), 0, 0], 
                [sin(theta), cos(theta), 0, 0], 
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    return T

def MDHTrans(alpha, a, d, theta):
    T=rotx(alpha)@trsfx(a)@trsfz(d)@rotz(theta)
    return T

def ForwardKinematics(q):
    q1, q2, q3, q4, q5, q6 = q
    T06 = np.zeros((4, 4))

    T06[0,0] = -((sin(q1)*sin(q4) - cos(q1)*cos(q4)*cos(q2 + q3))*cos(q5) + sin(q5)*sin(q2 + q3)*cos(q1))*cos(q6) - (sin(q1)*cos(q4) + sin(q4)*cos(q1)*cos(q2 + q3))*sin(q6)
    T06[0,1] = ((sin(q1)*sin(q4) - cos(q1)*cos(q4)*cos(q2 + q3))*cos(q5) + sin(q5)*sin(q2 + q3)*cos(q1))*sin(q6) - (sin(q1)*cos(q4) + sin(q4)*cos(q1)*cos(q2 + q3))*cos(q6)
    T06[0,2] = -(sin(q1)*sin(q4) - cos(q1)*cos(q4)*cos(q2 + q3))*sin(q5) + sin(q2 + q3)*cos(q1)*cos(q5)
    T06[0,3] = -0.07*sin(q1)*sin(q4)*sin(q5) - 0.02*sin(q1) + 0.4*sin(q2)*cos(q1) + 0.07*sin(q5)*cos(q1)*cos(q4)*cos(q2 + q3) + 0.07*sin(q2 + q3)*cos(q1)*cos(q5) + 0.45*sin(q2 + q3)*cos(q1)

    T06[1,0] = ((sin(q1)*cos(q4)*cos(q2 + q3) + sin(q4)*cos(q1))*cos(q5) - sin(q1)*sin(q5)*sin(q2 + q3))*cos(q6) - (sin(q1)*sin(q4)*cos(q2 + q3) - cos(q1)*cos(q4))*sin(q6)
    T06[1,1] = (-(sin(q1)*cos(q4)*cos(q2 + q3) + sin(q4)*cos(q1))*cos(q5) + sin(q1)*sin(q5)*sin(q2 + q3))*sin(q6) + (-sin(q1)*sin(q4)*cos(q2 + q3) + cos(q1)*cos(q4))*cos(q6)
    T06[1,2] = (sin(q1)*cos(q4)*cos(q2 + q3) + sin(q4)*cos(q1))*sin(q5) + sin(q1)*sin(q2 + q3)*cos(q5)
    T06[1,3] = 0.4*sin(q1)*sin(q2) + 0.07*sin(q1)*sin(q5)*cos(q4)*cos(q2 + q3) + 0.07*sin(q1)*sin(q2 + q3)*cos(q5) + 0.45*sin(q1)*sin(q2 + q3) + 0.07*sin(q4)*sin(q5)*cos(q1) + 0.02*cos(q1)

    T06[2,0] = -(sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*cos(q6) + sin(q4)*sin(q6)*sin(q2 + q3)
    T06[2,1] = (sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*sin(q6) + sin(q4)*sin(q2 + q3)*cos(q6)
    T06[2,2] = -sin(q5)*sin(q2 + q3)*cos(q4) + cos(q5)*cos(q2 + q3)
    T06[2,3] = -0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.4*cos(q2) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3)

    T06[3,0] = 0
    T06[3,1] = 0
    T06[3,2] = 0
    T06[3,3] = 1
    return T06

def JacobiWorld(q):
    q1, q2, q3, q4, q5, q6 = q
    J = np.zeros((6, 6))

    J[0,0] = -0.4*sin(q1)*sin(q2) - 0.07*sin(q1)*sin(q5)*cos(q4)*cos(q2 + q3) - 0.07*sin(q1)*sin(q2 + q3)*cos(q5) - 0.45*sin(q1)*sin(q2 + q3) - 0.07*sin(q4)*sin(q5)*cos(q1) - 0.02*cos(q1)
    J[0,1] = (-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.4*cos(q2) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3))*cos(q1)
    J[0,2] = (-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3))*cos(q1)
    J[0,3] = -0.07*(sin(q1)*cos(q4) + sin(q4)*cos(q1)*cos(q2 + q3))*sin(q5)
    J[0,4] = -0.07*sin(q1)*sin(q4)*cos(q5) - 0.07*sin(q5)*sin(q2 + q3)*cos(q1) + 0.07*cos(q1)*cos(q4)*cos(q5)*cos(q2 + q3)
    J[0,5] = 0

    J[1,0] = -0.07*sin(q1)*sin(q4)*sin(q5) - 0.02*sin(q1) + 0.4*sin(q2)*cos(q1) + 0.07*sin(q5)*cos(q1)*cos(q4)*cos(q2 + q3) + 0.07*sin(q2 + q3)*cos(q1)*cos(q5) + 0.45*sin(q2 + q3)*cos(q1)
    J[1,1] = (-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.4*cos(q2) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3))*sin(q1)
    J[1,2] = (-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3))*sin(q1)
    J[1,3] = 0.07*(-sin(q1)*sin(q4)*cos(q2 + q3) + cos(q1)*cos(q4))*sin(q5)
    J[1,4] = -0.07*sin(q1)*sin(q5)*sin(q2 + q3) + 0.07*sin(q1)*cos(q4)*cos(q5)*cos(q2 + q3) + 0.07*sin(q4)*cos(q1)*cos(q5)
    J[1,5] = 0

    J[2,0] = 0
    J[2,1] = -0.4*sin(q2) - 0.07*sin(q5)*cos(q4)*cos(q2 + q3) - 0.07*sin(q2 + q3)*cos(q5) - 0.45*sin(q2 + q3)
    J[2,2] = -0.07*sin(q5)*cos(q4)*cos(q2 + q3) - 0.07*sin(q2 + q3)*cos(q5) - 0.45*sin(q2 + q3)
    J[2,3] = 0.07*sin(q4)*sin(q5)*sin(q2 + q3)
    J[2,4] = -0.07*sin(q5)*cos(q2 + q3) - 0.07*sin(q2 + q3)*cos(q4)*cos(q5)
    J[2,5] = 0

    J[3,0] = 0
    J[3,1] = -sin(q1)
    J[3,2] = -sin(q1)
    J[3,3] = sin(q2 + q3)*cos(q1)
    J[3,4] = -sin(q1)*cos(q4) - sin(q4)*cos(q1)*cos(q2 + q3)
    J[3,5] = -sin(q1)*sin(q4)*sin(q5) + sin(q5)*cos(q1)*cos(q4)*cos(q2 + q3) + sin(q2 + q3)*cos(q1)*cos(q5)

    J[4,0] = 0
    J[4,1] = cos(q1)
    J[4,2] = cos(q1)
    J[4,3] = sin(q1)*sin(q2 + q3)
    J[4,4] = -sin(q1)*sin(q4)*cos(q2 + q3) + cos(q1)*cos(q4)
    J[4,5] = sin(q1)*sin(q5)*cos(q4)*cos(q2 + q3) + sin(q1)*sin(q2 + q3)*cos(q5) + sin(q4)*sin(q5)*cos(q1)

    J[5,0] = 1
    J[5,1] = 0
    J[5,2] = 0
    J[5,3] = cos(q2 + q3)
    J[5,4] = sin(q4)*sin(q2 + q3)
    J[5,5] = -sin(q5)*sin(q2 + q3)*cos(q4) + cos(q5)*cos(q2 + q3)
    return J

def JacobiLocal(angles):
    J = JacobiWorld(angles)
    T06 = ForwardKinematics(angles)
    R = T06[:3, :3]
    RR = block_diag(R, R)
    J = RR.T @ J
    return J

def dJacobiWorld(q, dq):
    q1, q2, q3, q4, q5, q6 = q
    dq1, dq2, dq3, dq4, dq5, dq6 = dq
    dJ = np.zeros((6, 6))
    
    dJ[0,0] = -dq1*(-0.07*sin(q1)*sin(q4)*sin(q5) - 0.02*sin(q1) + 0.4*sin(q2)*cos(q1) + 0.07*sin(q5)*cos(q1)*cos(q4)*cos(q2 + q3) + 0.07*sin(q2 + q3)*cos(q1)*cos(q5) + 0.45*sin(q2 + q3)*cos(q1)) - dq2*(-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.4*cos(q2) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3))*sin(q1) - dq3*(-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3))*sin(q1) + 0.07*dq4*(sin(q1)*sin(q4)*cos(q2 + q3) - cos(q1)*cos(q4))*sin(q5) - 0.07*dq5*(-sin(q1)*sin(q5)*sin(q2 + q3) + sin(q1)*cos(q4)*cos(q5)*cos(q2 + q3) + sin(q4)*cos(q1)*cos(q5))
    dJ[0,1] = -dq1*(-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.4*cos(q2) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3))*sin(q1) - dq2*(0.4*sin(q2) + 0.07*sin(q5)*cos(q4)*cos(q2 + q3) + 0.07*sin(q2 + q3)*cos(q5) + 0.45*sin(q2 + q3))*cos(q1) - dq3*(0.07*sin(q5)*cos(q4)*cos(q2 + q3) + 0.07*sin(q2 + q3)*cos(q5) + 0.45*sin(q2 + q3))*cos(q1) + 0.07*dq4*sin(q4)*sin(q5)*sin(q2 + q3)*cos(q1) - 0.07*dq5*(sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*cos(q1)
    dJ[0,2] = -dq1*(-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3))*sin(q1) - dq2*(0.07*sin(q5)*cos(q4)*cos(q2 + q3) + 0.07*sin(q2 + q3)*cos(q5) + 0.45*sin(q2 + q3))*cos(q1) - dq3*(0.07*sin(q5)*cos(q4)*cos(q2 + q3) + 0.07*sin(q2 + q3)*cos(q5) + 0.45*sin(q2 + q3))*cos(q1) + 0.07*dq4*sin(q4)*sin(q5)*sin(q2 + q3)*cos(q1) - 0.07*dq5*(sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*cos(q1)
    dJ[0,3] = 0.07*dq1*(sin(q1)*sin(q4)*cos(q2 + q3) - cos(q1)*cos(q4))*sin(q5) + 0.07*dq2*sin(q4)*sin(q5)*sin(q2 + q3)*cos(q1) + 0.07*dq3*sin(q4)*sin(q5)*sin(q2 + q3)*cos(q1) + 0.07*dq4*(sin(q1)*sin(q4) - cos(q1)*cos(q4)*cos(q2 + q3))*sin(q5) - 0.07*dq5*(sin(q1)*cos(q4) + sin(q4)*cos(q1)*cos(q2 + q3))*cos(q5)
    dJ[0,4] = -0.07*dq1*(-sin(q1)*sin(q5)*sin(q2 + q3) + sin(q1)*cos(q4)*cos(q5)*cos(q2 + q3) + sin(q4)*cos(q1)*cos(q5)) - 0.07*dq2*(sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*cos(q1) - 0.07*dq3*(sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*cos(q1) - 0.07*dq4*(sin(q1)*cos(q4) + sin(q4)*cos(q1)*cos(q2 + q3))*cos(q5) - 0.07*dq5*(-sin(q1)*sin(q4)*sin(q5) + sin(q5)*cos(q1)*cos(q4)*cos(q2 + q3) + sin(q2 + q3)*cos(q1)*cos(q5))
    dJ[0,5] = 0

    dJ[1,0] = -dq1*(0.4*sin(q1)*sin(q2) + 0.07*sin(q1)*sin(q5)*cos(q4)*cos(q2 + q3) + 0.07*sin(q1)*sin(q2 + q3)*cos(q5) + 0.45*sin(q1)*sin(q2 + q3) + 0.07*sin(q4)*sin(q5)*cos(q1) + 0.02*cos(q1)) + dq2*(-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.4*cos(q2) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3))*cos(q1) + dq3*(-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3))*cos(q1) - 0.07*dq4*(sin(q1)*cos(q4) + sin(q4)*cos(q1)*cos(q2 + q3))*sin(q5) - 0.07*dq5*(sin(q1)*sin(q4)*cos(q5) + sin(q5)*sin(q2 + q3)*cos(q1) - cos(q1)*cos(q4)*cos(q5)*cos(q2 + q3))
    dJ[1,1] = dq1*(-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.4*cos(q2) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3))*cos(q1) - dq2*(0.4*sin(q2) + 0.07*sin(q5)*cos(q4)*cos(q2 + q3) + 0.07*sin(q2 + q3)*cos(q5) + 0.45*sin(q2 + q3))*sin(q1) - dq3*(0.07*sin(q5)*cos(q4)*cos(q2 + q3) + 0.07*sin(q2 + q3)*cos(q5) + 0.45*sin(q2 + q3))*sin(q1) + 0.07*dq4*sin(q1)*sin(q4)*sin(q5)*sin(q2 + q3) - 0.07*dq5*(sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*sin(q1)
    dJ[1,2] = dq1*(-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3))*cos(q1) - dq2*(0.07*sin(q5)*cos(q4)*cos(q2 + q3) + 0.07*sin(q2 + q3)*cos(q5) + 0.45*sin(q2 + q3))*sin(q1) - dq3*(0.07*sin(q5)*cos(q4)*cos(q2 + q3) + 0.07*sin(q2 + q3)*cos(q5) + 0.45*sin(q2 + q3))*sin(q1) + 0.07*dq4*sin(q1)*sin(q4)*sin(q5)*sin(q2 + q3) - 0.07*dq5*(sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*sin(q1)
    dJ[1,3] = -0.07*dq1*(sin(q1)*cos(q4) + sin(q4)*cos(q1)*cos(q2 + q3))*sin(q5) + 0.07*dq2*sin(q1)*sin(q4)*sin(q5)*sin(q2 + q3) + 0.07*dq3*sin(q1)*sin(q4)*sin(q5)*sin(q2 + q3) - 0.07*dq4*(sin(q1)*cos(q4)*cos(q2 + q3) + sin(q4)*cos(q1))*sin(q5) - 0.07*dq5*(sin(q1)*sin(q4)*cos(q2 + q3) - cos(q1)*cos(q4))*cos(q5)
    dJ[1,4] = -0.07*dq1*(sin(q1)*sin(q4)*cos(q5) + sin(q5)*sin(q2 + q3)*cos(q1) - cos(q1)*cos(q4)*cos(q5)*cos(q2 + q3)) - 0.07*dq2*(sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*sin(q1) - 0.07*dq3*(sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*sin(q1) - 0.07*dq4*(sin(q1)*sin(q4)*cos(q2 + q3) - cos(q1)*cos(q4))*cos(q5) - 0.07*dq5*(sin(q1)*sin(q5)*cos(q4)*cos(q2 + q3) + sin(q1)*sin(q2 + q3)*cos(q5) + sin(q4)*sin(q5)*cos(q1))
    dJ[1,5] = 0

    dJ[2,0] = 0
    dJ[2,1] = -dq2*(-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.4*cos(q2) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3)) - dq3*(-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3)) + 0.07*dq4*sin(q4)*sin(q5)*cos(q2 + q3) + 0.07*dq5*(sin(q5)*sin(q2 + q3) - cos(q4)*cos(q5)*cos(q2 + q3))
    dJ[2,2] = -dq2*(-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3)) - dq3*(-0.07*sin(q5)*sin(q2 + q3)*cos(q4) + 0.07*cos(q5)*cos(q2 + q3) + 0.45*cos(q2 + q3)) + 0.07*dq4*sin(q4)*sin(q5)*cos(q2 + q3) + 0.07*dq5*(sin(q5)*sin(q2 + q3) - cos(q4)*cos(q5)*cos(q2 + q3))
    dJ[2,3] = 0.07*dq2*sin(q4)*sin(q5)*cos(q2 + q3) + 0.07*dq3*sin(q4)*sin(q5)*cos(q2 + q3) + 0.07*dq4*sin(q5)*sin(q2 + q3)*cos(q4) + 0.07*dq5*sin(q4)*sin(q2 + q3)*cos(q5)
    dJ[2,4] = 0.07*dq2*(sin(q5)*sin(q2 + q3) - cos(q4)*cos(q5)*cos(q2 + q3)) + 0.07*dq3*(sin(q5)*sin(q2 + q3) - cos(q4)*cos(q5)*cos(q2 + q3)) + 0.07*dq4*sin(q4)*sin(q2 + q3)*cos(q5) + 0.07*dq5*(sin(q5)*sin(q2 + q3)*cos(q4) - cos(q5)*cos(q2 + q3))
    dJ[2,5] = 0

    dJ[3,0] = 0
    dJ[3,1] = -dq1*cos(q1)
    dJ[3,2] = -dq1*cos(q1)
    dJ[3,3] = -dq1*sin(q1)*sin(q2 + q3) + dq2*cos(q1)*cos(q2 + q3) + dq3*cos(q1)*cos(q2 + q3)
    dJ[3,4] = dq1*(sin(q1)*sin(q4)*cos(q2 + q3) - cos(q1)*cos(q4)) + dq2*sin(q4)*sin(q2 + q3)*cos(q1) + dq3*sin(q4)*sin(q2 + q3)*cos(q1) + dq4*(sin(q1)*sin(q4) - cos(q1)*cos(q4)*cos(q2 + q3))
    dJ[3,5] = -dq1*(sin(q1)*sin(q5)*cos(q4)*cos(q2 + q3) + sin(q1)*sin(q2 + q3)*cos(q5) + sin(q4)*sin(q5)*cos(q1)) - dq2*(sin(q5)*sin(q2 + q3)*cos(q4) - cos(q5)*cos(q2 + q3))*cos(q1) - dq3*(sin(q5)*sin(q2 + q3)*cos(q4) - cos(q5)*cos(q2 + q3))*cos(q1) - dq4*(sin(q1)*cos(q4) + sin(q4)*cos(q1)*cos(q2 + q3))*sin(q5) - dq5*(sin(q1)*sin(q4)*cos(q5) + sin(q5)*sin(q2 + q3)*cos(q1) - cos(q1)*cos(q4)*cos(q5)*cos(q2 + q3))

    dJ[4,0] = 0
    dJ[4,1] = -dq1*sin(q1)
    dJ[4,2] = -dq1*sin(q1)
    dJ[4,3] = dq1*sin(q2 + q3)*cos(q1) + dq2*sin(q1)*cos(q2 + q3) + dq3*sin(q1)*cos(q2 + q3)
    dJ[4,4] = -dq1*(sin(q1)*cos(q4) + sin(q4)*cos(q1)*cos(q2 + q3)) + dq2*sin(q1)*sin(q4)*sin(q2 + q3) + dq3*sin(q1)*sin(q4)*sin(q2 + q3) - dq4*(sin(q1)*cos(q4)*cos(q2 + q3) + sin(q4)*cos(q1))
    dJ[4,5] = dq1*(-sin(q1)*sin(q4)*sin(q5) + sin(q5)*cos(q1)*cos(q4)*cos(q2 + q3) + sin(q2 + q3)*cos(q1)*cos(q5)) - dq2*(sin(q5)*sin(q2 + q3)*cos(q4) - cos(q5)*cos(q2 + q3))*sin(q1) - dq3*(sin(q5)*sin(q2 + q3)*cos(q4) - cos(q5)*cos(q2 + q3))*sin(q1) - dq4*(sin(q1)*sin(q4)*cos(q2 + q3) - cos(q1)*cos(q4))*sin(q5) + dq5*(-sin(q1)*sin(q5)*sin(q2 + q3) + sin(q1)*cos(q4)*cos(q5)*cos(q2 + q3) + sin(q4)*cos(q1)*cos(q5))

    dJ[5,0] = 0
    dJ[5,1] = 0
    dJ[5,2] = 0
    dJ[5,3] = (-dq2 - dq3)*sin(q2 + q3)
    dJ[5,4] = dq2*sin(q4)*cos(q2 + q3) + dq3*sin(q4)*cos(q2 + q3) + dq4*sin(q2 + q3)*cos(q4)
    dJ[5,5] = -dq2*(sin(q5)*cos(q4)*cos(q2 + q3) + sin(q2 + q3)*cos(q5)) - dq3*(sin(q5)*cos(q4)*cos(q2 + q3) + sin(q2 + q3)*cos(q5)) + dq4*sin(q4)*sin(q5)*sin(q2 + q3) - dq5*(sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))
    return dJ

def inverse_velocity_gram_schmidt(J, x_dot):
    J = np.array(J, dtype=float)
    x_dot = np.array(x_dot, dtype=float).flatten()
    
    n = J.shape[0]  # 自由度
    
    # ============ 第一步：Gram-Schmidt 正交化，构造 n_i ============
    n_vecs = []  # 存储正交向量 n_1, n_2, ..., n_n
    
    for i in range(n):
        # 取 J 的第 i 列 (0-based 索引)
        v = J[:, i].copy()
        
        # 减去在前面所有正交向量上的投影
        for j in range(i):
            # 投影系数: (n_j^T * v) / (n_j^T * n_j)
            proj_coeff = np.dot(n_vecs[j], v) / np.dot(n_vecs[j], n_vecs[j])
            v = v - proj_coeff * n_vecs[j]
        
        n_vecs.append(v)
    
    # ============ 第二步：反向回代求解 q_dot ============
    print(n_vecs)
    q_dot = np.zeros(n)
    residual = x_dot.copy()  # 当前剩余项: x_dot - sum_{j=i+1}^n J_j * q_dot_j
    
    # 从最后一个关节开始往前解 (i = n-1, n-2, ..., 0)
    for i in range(n-1, -1, -1):
        # 公式: q_dot_i = (residual^T * n_i) / (J_i^T * n_i)
        numerator = np.dot(residual, n_vecs[i])
        denominator = np.dot(J[:, i], n_vecs[i])
        
        # 防止除以零
        if abs(denominator) < 1e-12:
            raise ValueError(f"分母为零！J_{i+1}^T * n_{i+1} = 0，矩阵可能奇异。")
        
        q_dot[i] = numerator / denominator
        
        # 更新 residual: 减去已经求出的部分
        residual = residual - J[:, i] * q_dot[i]
    
    return q_dot


# ============ 运行测试 ============
if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=10, linewidth=200)
    q = np.random.uniform(-1.5, 1.5, 6)
    dq = np.random.uniform(-1, 1, 6)
    J = JacobiLocal(q)
    dx = J @ dq.reshape(6,1)
    print("关节速度:\n", dq)
    # 求解
    time0 = time.perf_counter()
    dq1 = inverse_velocity_gram_schmidt(J, dx)
    time1 = time.perf_counter()
    dq2 = np.linalg.inv(J) @ dx
    time2 = time.perf_counter()
    print("时间1:", time1 - time0, "矩阵QR关节速度:\n", dq1)
    print("时间2:", time2 - time1, "矩阵逆关节速度:\n", dq2)