import numpy as np
import math

def euler_to_matrix(rx, ry, rz, order='xyz'):
    """
    欧拉角转旋转矩阵
    order='xyz': R = Rz @ Ry @ Rx
    order='zyx': R = Rx @ Ry @ Rz
    """
    # 绕X轴
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    # 绕Y轴
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    # 绕Z轴
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0],
                   [0, 0, 1]])
    
    if order == 'xyz':
        return Rz @ Ry @ Rx
    else:  # zyx
        return Rx @ Ry @ Rz


def matrix_to_euler(R, order='xyz', degrees=False):
    """
    旋转矩阵转欧拉角
    order='xyz': 返回 (rx, ry, rz)
    order='zyx': 返回 (rz, ry, rx)
    """
    r11, r12, r13 = R[0,0], R[0,1], R[0,2]
    r21, r22, r23 = R[1,0], R[1,1], R[1,2]
    r31, r32, r33 = R[2,0], R[2,1], R[2,2]
    
    if order == 'xyz':
        # R = Rz @ Ry @ Rx
        ry = math.atan2(-r31, math.sqrt(r11**2 + r21**2))
        if abs(abs(ry) - math.pi/2) < 1e-6:
            rx = 0
            rz = math.atan2(r12, r22) if ry > 0 else math.atan2(-r12, -r22)
        else:
            rx = math.atan2(r32, r33)
            rz = math.atan2(r21, r11)
        result = (rx, ry, rz)
        
    else:  # zyx
        # R = Rx @ Ry @ Rz
        # 注意：这里用 r13 和 r23，不是 r11 和 r21
        ry = math.atan2(r13, math.sqrt(r23**2 + r33**2))  # 修正：用 r13 而不是 -r31
        
        if abs(abs(ry) - math.pi/2) < 1e-6:
            rz = 0
            rx = math.atan2(r21, r22) if ry > 0 else math.atan2(-r21, -r22)
        else:
            rx = math.atan2(-r23, r33)   # 修正
            rz = math.atan2(-r12, r11)   # 修正
        result = (rz, ry, rx)
    
    if degrees:
        return tuple(math.degrees(a) for a in result)
    return result


# 测试
if __name__ == "__main__":
    rx, ry, rz = math.radians(20), math.radians(60), math.radians(45)

    # XYZ顺序
    print("XYZ顺序:")
    R = euler_to_matrix(rx, ry, rz, 'xyz')
    print(R)
    print(matrix_to_euler(R, 'xyz', degrees=True))
    
    # ZYX顺序
    print("\nZYX顺序:")
    R = euler_to_matrix(rx, ry, rz, 'zyx')
    print(R)
    print(matrix_to_euler(R, 'zyx', degrees=True))