import cv2
from os import path
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pythonosc import udp_client
from face_ext.tongue_mask import is_have_tongue, is_have_tongue_

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QFont, QBrush, QLinearGradient
from PySide6.QtCore import QTimer, Qt, QSize, QDateTime, QPoint, QRect

# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 初始化 MediaPipe FaceLandmarker
model_path = path.abspath('face_ext/models/face_landmarker.task')
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

class CameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()


        self.setWindowFlags(Qt.FramelessWindowHint)  # 隐藏默认标题栏
        self.titleBar = CustomTitleBar(self)
        self.setMenuWidget(self.titleBar)  # 将自定义标题栏设置为菜单栏位置

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        # 设置整个窗口的背景颜色
        self.centralWidget.setStyleSheet("background-color: #5c81ca;")

        self.setGeometry(50, 0, 400, 600)
        self.move(50, 10)  # 设置窗口的位置


        # self.setWindowTitle("VRCDU V3.5")
        # self.setGeometry(100, 100, 400, 600)

        # 禁用最大化按钮
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        # 设置窗口的大小策略为固定大小
        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.original_size = self.size()

        # 初始化 OSC 客户端（初始值）
        self.osc_ip = "127.0.0.1"
        self.osc_port = 8888
        self.client = udp_client.SimpleUDPClient(self.osc_ip, self.osc_port)

        # 初始化 UI
        self.init_ui()
        
        # 获取可用相机
        self.available_cameras = self.get_available_cameras()
        
        # 如果没有可用的摄像头，就退出
        if not self.available_cameras:
            print("No cameras found")
            sys.exit()
            
        # 默认打开第一个摄像头
        self.current_camera = 0
        self.current_camera_name = None
        self.cap = cv2.VideoCapture(self.current_camera) # 打开摄像头
        # self.cap = cv2.VideoCapture("test_data/vrtest.mp4")  # 使用视频文件的人脸动作，替换为你的 MP4 文件路径
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # width = 640
        # height = 480
        # print(f"宽度: {width}")
        # print(f"高度: {height}")
        self.setGeometry(100, 100, width, height)
        if not self.cap.isOpened():
            print(f"Failed to open camera {self.current_camera}")
            sys.exit()

        # 启动定时器，每30ms更新一次视频帧
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # 初始化遮罩
        self.init_mask()

        # 启动时钟更新定时器
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)  # 每秒更新一次

    def init_ui(self):
        # 创建中央窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("background-color: #5c81ca;")
        
        # 创建垂直布局
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        # 创建视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)
        
        # 创建顶部水平布局（包含摄像机选择）
        self.camera_layout = QHBoxLayout()
        self.camera_container = QWidget()  # 创建一个容器来装摄像机选择布局
        self.camera_container.setLayout(self.camera_layout)
        # self.camera_container.setStyleSheet("background-color: rgba(0, 0, 0, 0);")  # 设置透明背景
        
        # 创建下拉框选择相机
        # 创建 QLabel 并设置字体颜色
        self.osc_camera_label = QLabel("Camera:")
        self.osc_camera_label.setStyleSheet("color: #FFFFFF;")  # 将字体颜色设置为白色
        self.camera_combo = QComboBox()
        # 设置 QComboBox 为平面样式，有边框，字体颜色为白色
        self.camera_combo.setStyleSheet("""
            QComboBox {
                color: #FFFFFF; /* 字体颜色为白色 */
                border: 1px solid #FFFFFF; /* 边框为白色，宽度为1像素 */
                background: #5c81ca; /* none为无背景 */
                font-size: 12px; /* 字体大小 */
                padding: 1px; /* 内边距 */
            }
            QComboBox:hover {
                border-color: #CCCCCC; /* 鼠标悬停时边框颜色变为浅灰色 */
            }
            QComboBox:focus {
                border-color: #FFFFFF; /* #999999聚焦时边框颜色变为深灰色 */
            }
            QComboBox::drop-down {
                border: none; /* 下拉按钮无边框 */
                background: none; /* 下拉按钮无背景 */
            }
            QComboBox::down-arrow {
                image: url(path/to/arrow.png); /* 自定义下拉箭头图标 */
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                color: #FFFFFF; /* 下拉菜单中的字体颜色为白色 */
                background: #5c81ca; /* 下拉菜单的背景颜色 */
                border: 1px solid #FFFFFF; /* 下拉菜单的边框颜色 */
                selection-color: #FFFFFF; /* 选中项的字体颜色 */
                selection-background-color: #5c81ca; /* 选中项的背景颜色 */
            }
        """)
        self.camera_combo.setFixedWidth(210)
        self.camera_layout.addWidget(self.osc_camera_label)
        self.camera_layout.addWidget(self.camera_combo)
        
        # # 创建 OSC 布局（包含 OSC IP 和端口）
        # self.osc_layout = QHBoxLayout()
        # self.osc_container = QWidget()  # 创建一个容器来装 OSC 布局
        # self.osc_container.setLayout(self.osc_layout)
        # # self.osc_container.setStyleSheet("background-color: rgba(0, 0, 0, 0);")  # 设置透明背景
        
        # # 创建文本框输入 OSC IP 和端口
        # self.osc_layout.addWidget(QLabel("OSC IP:"))
        # self.osc_ip_edit = QLineEdit(self.osc_ip)
        # self.osc_layout.addWidget(self.osc_ip_edit)
        
        # self.osc_layout.addWidget(QLabel("OSC Port:"))
        # self.osc_port_edit = QLineEdit(str(self.osc_port))
        # self.osc_port_edit.setFixedWidth(60)
        # self.osc_layout.addWidget(self.osc_port_edit)

        # 创建 QLabel 并设置字体颜色
        self.osc_ip_label = QLabel("OSC IP:")
        self.osc_ip_label.setStyleSheet("color: #FFFFFF;")  # 将字体颜色设置为白色
        self.camera_layout.addWidget(self.osc_ip_label)
        self.osc_ip_edit = QLineEdit(self.osc_ip)
        # 设置文本框为平面样式，有边框，字体颜色为白色
        self.osc_ip_edit.setStyleSheet("""
            QLineEdit {
                color: #FFFFFF; /* 字体颜色为白色 */
                border: 1px solid #FFFFFF; /* 边框为白色，宽度为1像素 */
                background: #5c81ca; /* none为无背景 */
                font-size: 12px; /* 字体大小 */
                padding: 1px; /* 内边距 */
            }
            QLineEdit:hover {
                border-color: #CCCCCC; /* 鼠标悬停时边框颜色变为浅灰色 */
            }
            QLineEdit:focus {
                border-color: #FFFFFF; /* #999999聚焦时边框颜色变为深灰色 */
            }
        """)
        self.camera_layout.addWidget(self.osc_ip_edit)

        # 创建 QLabel 并设置字体颜色
        self.osc_port_label = QLabel("OSC Port:")
        self.osc_port_label.setStyleSheet("color: #FFFFFF;")  # 将字体颜色设置为白色
        self.camera_layout.addWidget(self.osc_port_label)
        self.osc_port_edit = QLineEdit(str(self.osc_port))
        # 设置文本框为平面样式，有边框，字体颜色为白色
        self.osc_port_edit.setStyleSheet("""
            QLineEdit {
                color: #FFFFFF; /* 字体颜色为白色 */
                border: 1px solid #FFFFFF; /* 边框为白色，宽度为1像素 */
                background: #5c81ca; /* none为无背景 */
                font-size: 12px; /* 字体大小 */
                padding: 1px; /* 内边距 */
            }
            QLineEdit:hover {
                border-color: #CCCCCC; /* 鼠标悬停时边框颜色变为浅灰色 */
            }
            QLineEdit:focus {
                border-color: #FFFFFF; /* #999999聚焦时边框颜色变为深灰色 */
            }
        """)
        self.osc_port_edit.setFixedWidth(70)
        self.camera_layout.addWidget(self.osc_port_edit)
        
        # 创建控制按钮水平布局
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)
        
        # 创建控制按钮
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.setStyleSheet("""
            QPushButton {
                color: #FFFFFF; /* 字体颜色为白色 */
                border: 1px solid #FFFFFF; /* 边框为白色，宽度为1像素 */
                background: none; /* 无背景 */
                font-size: 12px; /* 字体大小 */
            }
            QPushButton:hover {
                color: #CCCCCC; /* 鼠标悬停时字体颜色变为浅灰色 */
            }
            QPushButton:pressed {
                color: #999999; /* 按下时字体颜色变为深灰色 */
            }
        """)
        self.stop_button.setStyleSheet("""
            QPushButton {
                color: #FFFFFF; /* 字体颜色为白色 */
                border: 1px solid #FFFFFF; /* 边框为白色，宽度为1像素 */
                background: none; /* 无背景 */
                font-size: 12px; /* 字体大小 */
            }
            QPushButton:hover {
                color: #CCCCCC; /* 鼠标悬停时字体颜色变为浅灰色 */
            }
            QPushButton:pressed {
                color: #999999; /* 按下时字体颜色变为深灰色 */
            }
        """)
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)
        
        # 连接按钮点击事件
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

    def get_available_cameras(self):
        # available_cameras = []
        # for i in range(10):  # 检查前10个摄像头
        #     cap = cv2.VideoCapture(i)
        #     if cap.isOpened():
        #         available_cameras.append(str(i))
        #         cap.release()
        # return available_cameras

        from PyCameraList.camera_device import list_video_devices
        available_cameras = []
        cameras = list_video_devices()
        for index, camera_name  in cameras:  
            available_cameras.append(f"{index}. {camera_name}")
        return available_cameras

    def update_camera_combo(self):
        self.camera_combo.clear()
        self.camera_combo.addItems(self.available_cameras)
        self.camera_combo.setCurrentIndex(0)
        self.current_camera_name = self.available_cameras[0].split('.')[1].strip()
        # 连接下拉框选择事件
        self.camera_combo.currentIndexChanged.connect(self.change_camera)

    def init_mask(self):
        # 创建一个灰色遮罩层
        self.mask = QLabel(self.video_label)
        # self.mask.setStyleSheet("background-color: rgba(128, 128, 128, 255);")  # 设置为灰色
        # self.mask.setStyleSheet("background-color: rgba(128, 128, 128, 200);")  # 设置为灰色并调整透明度
        self.mask.setStyleSheet("background-color: #5c81ca;")
        self.mask.setAlignment(Qt.AlignTop)
        self.mask.resize(self.video_label.size().width(), self.video_label.size().height()*2 // 3)
        self.mask.show()

        # 创建一个时钟标签并将其添加到遮罩层上
        self.clock_label = QLabel(self.mask)
        self.clock_label.setAlignment(Qt.AlignCenter)  # 时钟居中
        self.clock_label.setGeometry(0, 0, self.mask.width(), self.mask.height())  # 设置时钟标签大小

        # 设置字体大小
        font = QFont()
        font.setPointSize(59)  # 设置字体大小为49
        self.clock_label.setFont(font)

        # 设置文字颜色为绿色
        self.clock_label.setStyleSheet("color: green;")

        # 设置时钟格式
        self.update_clock()  # 初始设置时钟显示

        # 将摄像机选择布局添加到视频标签的左上角
        self.camera_container.setParent(self.video_label)
        self.camera_container.move(0, 0)  # 设置摄像机选择布局的位置
        self.camera_container.show()

        # # 将 OSC 布局添加到摄像机选择布局的下方
        # self.osc_container.setParent(self.video_label)
        # self.osc_container.move(0, 30)  # 设置 OSC 布局的位置
        # self.osc_container.show()

    def update_clock(self):
        # 获取当前时间
        current_time = QDateTime.currentDateTime()
        # 格式化为小时:分钟
        time_str = current_time.toString("hh:mm:ss")
        # 更新时钟标签显示
        self.clock_label.setText(time_str)

    def update_frame(self):
        # 读取一帧
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return
        
        # 图像处理部分保持不变
        height, width = frame.shape[:2]
        # 获取图像的下半部分
        lower_half = frame[height//2:, :]
        # frame = lower_half
        # 上部分
        up = cv2.imread('face_ext/01.png')  # 读取上部分图像
        up = cv2.resize(up, (width, height*2//3))  # 调整大小
        # up = cv2.resize(up, (width, height))
        down = frame
        
        # 将上半部分与下半部分合并
        if "virtual" not in self.current_camera_name.lower():
            frame = np.vstack((up, down))
        
        # 将图像从 BGR 转换为 RGB
        image_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # 检测面部地标和表情参数
        detection_result = detector.detect(mp_image)

        # 绘制面部地标并显示表情参数
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                # 绘制面部地标
                for landmark in face_landmarks:
                    x = int(frame.shape[1] - landmark.x * frame.shape[1])  # 调整 x 坐标
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # 发送表情参数到 OSC
            if detection_result.face_blendshapes:
                for category in detection_result.face_blendshapes:
                    for blendshape in category:
                        # mouthLowerDownLeft
                        # mouthLowerDownRight
                        # mouthSmileLeft
                        # mouthSmileRight
                        # mouthUpperUpLeft
                        # mouthUpperUpRight
                        self.client.send_message("/" + blendshape.category_name, blendshape.score)
                        # print(f"表情动作: {blendshape.category_name}，{blendshape.score}")

            # 发送舌头伸出值
            avatar_tongue_out_value = round(is_have_tongue_(frame, face_landmarks), 2)
            self.client.send_message("/tongueOut", avatar_tongue_out_value)

        # 转换为 Qt 可显示的格式
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(640, 480+240, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

        # 调整遮罩大小
        self.adjust_mask_size()

    def adjust_mask_size(self):
        if self.video_label.pixmap():
            pixmap_size = self.video_label.pixmap().size()
            self.mask.resize(pixmap_size.width(), pixmap_size.height() // 2)
            # 调整时钟标签大小
            self.clock_label.setGeometry(0, 0, self.mask.width(), self.mask.height())
            # 调整摄像机选择和 OSC 布局的位置
            self.camera_container.resize(self.camera_container.sizeHint())
            self.camera_container.move(0, 0)
            # self.osc_container.resize(self.osc_container.sizeHint())
            # self.osc_container.move(0, 30)

    def change_camera(self, index):
        # 如果当前已经有开启的摄像头，先释放资源
        self.cap.release()
        # 打开新的摄像头
        # self.current_camera = int(self.available_cameras[index])
        self.current_camera = int(self.available_cameras[index].split('.')[0].strip())
        self.current_camera_name = self.available_cameras[index].split('.')[1].strip()
        self.cap = cv2.VideoCapture(self.current_camera)
        if not self.cap.isOpened():
            print(f"Failed to open camera {self.current_camera}")

    def start_camera(self):
        self.update_osc_client()  # 在启动相机前更新 OSC 客户端
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()

    def update_osc_client(self):
        # 更新 OSC 客户端
        try:
            new_ip = self.osc_ip_edit.text().strip()
            new_port = int(self.osc_port_edit.text().strip())
            self.osc_ip = new_ip
            self.osc_port = new_port
            self.client = udp_client.SimpleUDPClient(new_ip, new_port)
        except ValueError:
            print("Invalid OSC port value. Using default value 8888.")
            self.osc_port_edit.setText(str(self.osc_port))

    def closeEvent(self, event):
        # 释放摄像头资源
        self.cap.release()
        # 停止定时器
        self.timer.stop()
        self.clock_timer.stop()
        # 关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()
        # 关闭应用
        event.accept()

    def resizeEvent(self, event):
        # super().resizeEvent(event)
        # 重写 resizeEvent 方法，将窗口大小设置为初始大小
        self.resize(self.original_size)
        self.adjust_mask_size()

class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        # 设置整个标题栏的背景颜色
        self.setStyleSheet("background-color: #5c81ca;")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.titleLabel = QLabel("VRCDU 0.5.1beta")
        self.titleLabel.setStyleSheet("color: white; font-size: 16px; padding: 5px;")
        layout.addWidget(self.titleLabel)

        # 添加按钮布局
        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonLayout.setSpacing(0)
        self.buttonLayout.setAlignment(Qt.AlignRight)

        # 最小化按钮
        self.minimizeButton = QPushButton()
        self.minimizeButton.setFixedSize(30, 30)
        self.minimizeButton.setStyleSheet("""
        	QPushButton {
        	    background-color: #5c81ca;
        	    border: none;
        	    color: white;
        	    font-size: 18px;
        	    padding: 0;
        	}
        	QPushButton:hover {
        	    background-color: #45a049;
        	}
        	QPushButton:pressed {
        	    background-color: red; /* 按下时的背景颜色设置为红色 */
        	}
        	""")
        self.minimizeButton.setText("–")
        self.minimizeButton.clicked.connect(self.parent.showMinimized)

        # 关闭按钮
        self.closeButton = QPushButton()
        self.closeButton.setFixedSize(30, 30)
        self.closeButton.setStyleSheet("""
        	QPushButton {
        	    background-color: #5c81ca;
        	    border: none;
        	    color: white;
        	    font-size: 18px;
        	    padding: 0;
        	}
        	QPushButton:hover {
        	    background-color: #d32f2f;
        	}
        	QPushButton:pressed {
        	    background-color: red; /* 按下时的背景颜色设置为红色 */
        	}
        	""")
        self.closeButton.setText("×")
        self.closeButton.clicked.connect(self.parent.close)

        self.buttonLayout.addWidget(self.minimizeButton)
        self.buttonLayout.addWidget(self.closeButton)

        layout.addLayout(self.buttonLayout)

        self.setLayout(layout)

    def mousePressEvent(self, event):
        self.parent.offset = event.pos()

    def mouseMoveEvent(self, event):
        x = event.globalPosition().x() - self.parent.offset.x()
        y = event.globalPosition().y() - self.parent.offset.y()
        self.parent.move(int(x), int(y))

    def paintEvent(self, event):
	    super().paintEvent(event)
	    painter = QPainter(self)
	    painter.setRenderHint(QPainter.Antialiasing)  # 启用抗锯齿

	    gradient = QLinearGradient(0, 0, self.width(), 0)
	    gradient.setColorAt(0, QColor("#5c81ca"))  # 左边红色
	    gradient.setColorAt(1, QColor("#5c81ca"))  # 右边完全透明

	    painter.setBrush(QBrush(gradient))
	    painter.setPen(Qt.NoPen)  # 设置画笔为无边界
	    painter.drawRect(QRect(0, 0, self.width(), self.height()+1))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraWindow()
    window.update_camera_combo()
    window.show()
    sys.exit(app.exec())
