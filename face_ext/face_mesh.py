import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# MediaPipe的关键点索引与OpenCV不同，需调整索引
LEFT_EYE_IDXS = [133, 159, 145, 144, 147]  # 左眼（MediaPipe索引与OpenCV一致）
RIGHT_EYE_IDXS = [362, 386, 374, 373, 376] # 右眼
# UPPER_LIP_IDXS = [61, 185, 40, 39, 37]     # 上唇上部（MediaPipe上唇顶部索引为61等）
# UPPER_LIP_IDXS = [185, 40, 39, 37, 0]     # 上唇上部（MediaPipe上唇顶部索引为61等）
# LOWER_LIP_IDXS = [146, 91, 181, 84, 17]    # 下唇下部

UPPER_LIP_IDXS = [191, 80, 81, 82, 13]     # 上唇下部
LOWER_LIP_IDXS = [95, 88, 178, 87, 14]    # 下唇上部

def calculate_eye_distance(face_landmarks, image_shape):
    left_eye_points = [face_landmarks.landmark[idx] for idx in LEFT_EYE_IDXS]
    left_x = np.mean([pt.x for pt in left_eye_points]) * image_shape[1]
    left_y = np.mean([pt.y for pt in left_eye_points]) * image_shape[0]
    
    right_eye_points = [face_landmarks.landmark[idx] for idx in RIGHT_EYE_IDXS]
    right_x = np.mean([pt.x for pt in right_eye_points]) * image_shape[1]
    right_y = np.mean([pt.y for pt in right_eye_points]) * image_shape[0]
    
    return np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)

def calculate_normalized_openness(image, face_landmarks):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # results = face_mesh.process(image_rgb)
    
    # if not results.multi_face_landmarks:
    #     return None, None, image
    
    # face_landmarks = results.multi_face_landmarks[0]
    h, w = image.shape[:2]
    
    # 计算张口原始距离
    upper_y = np.mean([face_landmarks.landmark[idx].y for idx in UPPER_LIP_IDXS]) * h
    lower_y = np.mean([face_landmarks.landmark[idx].y for idx in LOWER_LIP_IDXS]) * h
    raw_openness = lower_y - upper_y
    
    # 计算两眼间距
    eye_distance = calculate_eye_distance(face_landmarks, (h, w))
    
    if eye_distance == 0:
        return None, raw_openness, image
    
    normalized_openness = raw_openness / eye_distance
    
    # # 可视化（MediaPipe坐标需转换为像素）
    # for idx in UPPER_LIP_IDXS + LOWER_LIP_IDXS + LEFT_EYE_IDXS + RIGHT_EYE_IDXS:
    #     pt = face_landmarks.landmark[idx]
    #     x, y = int(pt.x * w), int(pt.y * h)
    #     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # cv2.putText(image, f"Normalized: {normalized_openness:.2f}", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return normalized_openness, raw_openness, image

def calculate_normalized_openness_(image, face_landmarks):
    normalized, raw, result = calculate_normalized_openness(image, face_landmarks)
    return normalized * 2


# 测试代码
if __name__ == "__main__":
    image = cv2.imread("../test_data/5.png")
    normalized, raw, result = calculate_normalized_openness(image)
    
    if normalized is not None:
        print(f"归一化张口大小：{normalized:.2f}")
        print(f"原始张口距离：{raw:.1f} 像素")
    else:
        print("检测失败")
    
    # cv2.imshow("Normalized Mouth Openness", result)
    # cv2.waitKey(0)


    image = cv2.imread("../test_data/6.png")
    normalized, raw, result = calculate_normalized_openness(image)
    
    if normalized is not None:
        print(f"归一化张口大小：{normalized:.2f}")
        print(f"原始张口距离：{raw:.1f} 像素")
    else:
        print("检测失败")



    image = cv2.imread("../test_data/7.png")
    normalized, raw, result = calculate_normalized_openness(image)
    
    if normalized is not None:
        print(f"归一化张口大小：{normalized:.2f}")
        print(f"原始张口距离：{raw:.1f} 像素")
    else:
        print("检测失败")

