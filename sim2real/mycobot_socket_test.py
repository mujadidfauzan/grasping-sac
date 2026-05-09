from pymycobot import MyCobotSocket

mc = MyCobotSocket("10.42.0.1", 9000)
# 172.17.9

# Kirim data sudut
mc.send_angles([0, 0, 0, 0, 0, 0], 50)
