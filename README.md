# face_tracking 
一个极简、开箱即用的 **实时面部捕捉 → OSC 转发** 小工具。  
基于 **MediaPipe** 提取 468 点面部 landmark + 52 组 ARKit blendshape，通过 **PySide6** 构建无边框透明窗口，把数据实时推送到 VRChat、Unity、Blender 等支持 OSC 的接收端。

---

## ✨ 功能
- ✅ 多摄像头热切换（含虚拟相机）  
- ✅ 52 组 blendshape 实时输出  
- ✅ 舌头伸出检测（自带 CNN）  
- ✅ 无边框可拖动窗口 + 固定比例  
- ✅ 动态修改 OSC IP / Port  
- ✅ 顶部半透明时钟遮罩  

## 🚧 计划实现（暂未支持）

| 功能 | 描述 |
|---|---|
| **舌头上下左右伸出检测** | 目前仅输出 0-1 的“伸出程度”，后续计划增加 `tongueUp`, `tongueDown`, `tongueLeft`, `tongueRight` 四维方向参数，实现 360° 舌头姿态捕捉。 |
| **开源训练模型** | 将数据预处理、模型训练开源。 |
---

## 🚀 一键运行
### 1. 克隆
```bash
git clone https://github.com/okgptai/face_tracking.git
cd face_tracking
```

### 2. 依赖
```bash
pip install -r requirements.txt
```

### 3. 下载模型
将官方 [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)  放入 `face_ext/models/` 文件夹。

### 4. 启动
```bash
python face_tracking_pyside6.py
```

---

## 📦 依赖
```
opencv-python>=4.8
mediapipe>=0.10
numpy
PySide6>=6.5
python-osc
PyCameraList
```

---

## 🔌 OSC 映射
| 地址 | 值域 | 说明 |
|---|---|---|
| `/tongueOut` | 0.0 – 1.0 | 舌头伸出程度 |
| `/browDownLeft` … `mouthSmileRight` | 0.0 – 1.0 | 52 组 ARKit 系数 |

默认目标：`127.0.0.1:8888`  
在 UI 左上角可直接修改 IP / Port，点击 **Start** 生效。

---

## 📁 目录
```
FaceTracking-OSC/
├── main.py                 # 主程序
├── face_ext/
│   ├── face.model          # 需自行下载
│   ├── tongue_mask.py      # 舌头检测
│   └── 01.png              # 顶部遮罩图
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🧪 舌头检测
`tongue_mask.py` 内置轻量 CNN，对口内区域做二分类，输出 0-1 连续值，可直接驱动 avatar 的 Tongue Out 参数。

---

## 🎮 VRChat 对接
1. Avatar 开启 **OSC Enabled**  
2. Parameters 名称与 blendshape 完全一致（区分大小写）  
3. Action Menu → **OSC Debug** 可实时查看数值

---

## 🛠️ 打包 exe
```bash
pip install pyinstaller
pyinstaller -F -w main.py --add-data "face_ext;face_ext" -n face_tracking
```
可执行文件在 `dist/face_tracking.exe`

---

## 📄 鸣谢
MediaPipe、Wav2Lip、ResNet

---

## 📄 许可证
MIT License
```
