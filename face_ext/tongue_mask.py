import os
import cv2
import mediapipe as mp

cascade_path = r"face_ext/models/tongue_cascade.xml"

if not os.path.exists(cascade_path):
    print(f"分类器文件不存在，请检查路径: {cascade_path}")
    exit()

custom_cascade = cv2.CascadeClassifier(cascade_path)
if custom_cascade.empty():
    print("分类器加载失败，请检查文件是否正确。")
    exit()

def is_have_tongue(frame):
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用 detectMultiScale 进行目标检测
    objects = custom_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return objects

def is_have_tongue_(frame, face_landmarks):
    ################画出舌头,舌头的上边在上嘴唇以下且舌头的下边在下嘴唇以下时，才画###################
    # 获取上嘴唇和下嘴唇的y坐标
    upper_lip_y = int(face_landmarks[0].y * frame.shape[0])  # 假设第0个关键点是上嘴唇
    lower_lip_y = int(face_landmarks[17].y * frame.shape[0])  # 假设第17个关键点是下嘴唇
    tongue_out_value = 0.0

    # 检测舌头
    objects = is_have_tongue(frame)

    # 绘制检测结果
    for (x, y, w, h) in objects:
        # 判断舌头的上边是否在上嘴唇以下，且舌头的下边是否在下嘴唇以下
        # if y > upper_lip_y and y + h < lower_lip_y:
        if y > upper_lip_y: 
            if y + h < lower_lip_y:
                tongue_out_value = 0.7
            if y + h >= lower_lip_y:
                tongue_out_value = 1.0
        # print('舌头的上边:', y)
        # print('上嘴唇下边:', upper_lip_y)
        # print('舌头的下边:', y + h)
        # print('上嘴唇上边:', lower_lip_y)
        # print('tongue_out_value:', tongue_out_value)

    return tongue_out_value
    ################画出舌头,舌头的上边在上嘴唇以下且舌头的下边在下嘴唇以下，才画###################


# 测试代码
if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # 使用更精细的关键点（包括舌头）
        min_detection_confidence=0.5,
    )

    # 舌头关键点索引（部分示例）
    TONGUE_INDICES = [61,  185, 40,  39,  37,  0,  267, 269, 270, 409, 
                      291, 375, 321, 405, 314, 17, 84,  181, 91, 146]

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # 转换为RGB并处理
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 提取关键点
                for idx in TONGUE_INDICES:
                    landmark = face_landmarks.landmark[idx]
                    h, w, c = image.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # 画绿色点

        # 显示结果
        cv2.imshow('Tongue Tracking', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        objects = is_have_tongue(image)

        # 设置窗口初始大小（可选）
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection", 800, 600)  # 设置窗口大小为 800x600

        # 绘制检测结果
        for (x, y, w, h) in objects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示结果

        cv2.imshow("Detection", image)



    cap.release()
    cv2.destroyAllWindows()