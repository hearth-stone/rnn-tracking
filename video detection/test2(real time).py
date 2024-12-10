import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# 初始化MediaPipe Hands模块
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 存储9号点的历史位置
landmark_history = []

# 设置Matplotlib图形
plt.ion()  # 交互模式
fig, ax = plt.subplots()
line, = ax.plot([], [], 'k-', linewidth=2)  # 黑色线，线宽为2
ax.set_xlim(0, 640)  # 假设图像宽度为640
ax.set_ylim(0, 480)  # 假设图像高度为480
plt.title('Hand Landmark Trajectory')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# 初始化Matplotlib图形的x和y数据
x_data, y_data = [], []

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if ret:
        # 实现镜像效果
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                for i, lm in enumerate(handLms.landmark):
                    if i == 9:
                        xPos = int(lm.x * img.shape[1])
                        yPos = int(lm.y * img.shape[0])
                        landmark_history.append((xPos, yPos))
                        if len(landmark_history) > 50:
                            landmark_history.pop(0)

                        # 更新Matplotlib图形的x和y数据
                        x_data.append(xPos)
                        y_data.append(yPos)
                        line.set_data(x_data, y_data)

                        # 重新绘制图形
                        ax.relim()  # 重新计算坐标轴限制
                        ax.autoscale_view()  # 自动缩放视图
                        fig.canvas.draw()
                        fig.canvas.flush_events()

        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 停止Matplotlib的交互模式并显示最终图形
plt.ioff()
plt.show()