import pinocchio as pin
import numpy as np

# 关闭科学计数法，保留 10 位小数
np.set_printoptions(suppress=True, precision=10, linewidth=200)
q = np.array([1, 1, 1, 1, 1, 1])
dq = np.array([0.5, 0.8, 1.1, 0.4, 0.2, 0.5])
ddq = np.array([0, 0.2, 0, 0.5, 0, 0])

# 从 URDF 加载机器人
model = pin.buildModelFromUrdf("sym\\six_axis_arm2.urdf")
data = model.createData()
pin.forwardKinematics(model, data, q, dq, ddq)
frame_id = model.getFrameId("6_Link")

RT06 = pin.updateFramePlacement(model, data, frame_id)
print("pin位置:\n", RT06.translation, "\n", RT06.rotation)

vLocal = pin.getFrameVelocity(model, data, frame_id, pin.ReferenceFrame.LOCAL)
print("pin速度:\n", vLocal.linear, "\n", vLocal.angular)

aLocal = pin.getFrameClassicalAcceleration(model, data, frame_id, pin.ReferenceFrame.LOCAL)
print("pin加速度:\n", aLocal.linear, "\n", aLocal.angular)

tau = pin.rnea(model, data, q, dq, ddq)
print("关节力矩:", tau)