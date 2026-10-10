import sympy as sp
from sympy import sin, cos, pi, simplify, Matrix, eye, diff
sp.init_printing()  # 让输出结果更美观

# 连杆扭转角alpha     连杆长度a       连杆偏距d       关节角theta
#       0               0               0               0
#       -90	            0	            0	            -90
#       0	            400	            20	            90
#       90	            0	            450	            0
#       -90	            0	            0	            0
#       90	            0	            70	            0

def MDHTrans(alpha, a, d, theta):
    """返回改进DH参数的齐次变换矩阵"""
    return sp.Matrix([
        [cos(theta), -sin(theta), 0, a],
        [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -d*sin(alpha)],
        [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), d*cos(alpha)],
        [0, 0, 0, 1]
    ])

def SDHTrans(d, theta, alpha, a,):
    """返回标准DH参数的齐次变换矩阵"""
    return sp.Matrix([
        [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
        [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])

sp.init_printing()  # 让输出结果更美观
q = sp.symbols('q1:7')
dq = sp.symbols('dq1:7')
q1, q2, q3, q4, q5, q6 = q[0], q[1], q[2], q[3], q[4], q[5]
dq1, dq2, dq3, dq4, dq5, dq6 = dq[0], dq[1], dq[2], dq[3], dq[4], dq[5]

if False:
    T01 = MDHTrans(    0,   0,    0, q[0]     )
    T12 = MDHTrans(-pi/2,   0,    0, q[1]-pi/2)
    T23 = MDHTrans(    0, 0.4, 0.02, q[2]+pi/2)
    T34 = MDHTrans( pi/2,   0, 0.45, q[3]     )
    T45 = MDHTrans(-pi/2,   0,    0, q[4]     )
    T56 = MDHTrans( pi/2,   0, 0.07, q[5]     )
    T06=T01*T12*T23*T34*T45*T56

    for i in range(4):
        for j in range(4):
            print(f"T06[{i},{j}] = {simplify(T06[i, j])}")
        print()
else:
    T06 = eye(4, 4)
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

if False:
    R=T06[:3, :3]
    J = Matrix.zeros(6, 6)
    for i in range(6):
        J[0,i] = diff(T06[0,3], q[i])
        J[1,i] = diff(T06[1,3], q[i])
        J[2,i] = diff(T06[2,3], q[i])
        W=simplify(R.diff(q[i]))*R.T
        J[3,i] = W[2,1]
        J[4,i] = W[0,2]
        J[5,i] = W[1,0]

    for i in range(6):
        for j in range(6):
            print(f"J[{i},{j}] = {simplify(J[i, j])}")
        print()
else:
    J = Matrix.zeros(6, 6)
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


dJ = Matrix.zeros(6, 6)
# 对每个关节变量求偏导并乘以对应的速度
for i in range(6):      # 行
    for j in range(6):  # 列
        # 对前5个关节变量求导（因为J不依赖t6）
        dJ[i,j] = sum(diff(J[i,j], q[k]) * dq[k] for k in range(5))
        # 如果J依赖t6，将range(5)改为range(6)
        print(f"dJ[{i},{j}] = {simplify(dJ[i, j])}")