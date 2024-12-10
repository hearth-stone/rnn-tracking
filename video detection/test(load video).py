import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe Hands模块
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# 存储9号点的历史位置
landmark_history = []

# 视频文件路径
cap = cv2.VideoCapture("C:/Users/U5716/PycharmProjects/handTracking/.venv/Lib/test/test1.mp4")

# 创建轨迹图
trajectory_img = np.ones((720, 1280, 3), dtype=np.uint8) * 255

def mouse_callback(event, x, y, flags, param):
    global landmark_history, trajectory_img

    # 当鼠标移动时显示坐标
    if event == cv2.EVENT_MOUSEMOVE:
        for point in landmark_history:
            if (x - point[0]) ** 2 + (y - point[1]) ** 2 < 100:  # 简化的接近度检测
                cv2.circle(trajectory_img, point, 5, (255, 0, 0), -1)  # 绘制一个小红点
                cv2.putText(trajectory_img, f"({point[0]}, {point[1]})", (point[0]+10, point[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.imshow('Trajectory', trajectory_img)
                break

# 设置鼠标回调函数
cv2.namedWindow('Trajectory')
cv2.setMouseCallback('Trajectory', mouse_callback)

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    # 绘制历史轨迹
    for i in range(1, len(landmark_history)):
        x1, y1 = landmark_history[-i][0], landmark_history[-i][1]
        x2, y2 = landmark_history[-(i + 1)][0], landmark_history[-(i + 1)][1]
        cv2.line(trajectory_img, (x1, y1), (x2, y2), (50, 50, 50), 4)  # 灰色线，线宽为4

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for i, lm in enumerate(handLms.landmark):
                xPos = int(lm.x * img.shape[1])
                yPos = int(lm.y * img.shape[0])
                cv2.putText(img, str(i), (xPos - 25, yPos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

                # 绘制9号点的运动轨迹
                if i == 9:
                    landmark_history.append((xPos, yPos))
                    # 限制历史点的数量，以避免内存溢出
                    if len(landmark_history) > 1000:
                        landmark_history.pop(0)

    cv2.imshow('Video', img)
    cv2.imshow('Trajectory', trajectory_img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 保持轨迹图片窗口打开，直到用户关闭它
while True:
    cv2.imshow('Trajectory', trajectory_img)
    if cv2.waitKey(1) == ord('q'):
        break