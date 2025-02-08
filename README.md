# 一画室智能图片命名工具 v4.5

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-green)
![OCR](https://img.shields.io/badge/OCR-EasyOCR%20%26%20PaddleOCR-orange)

一款基于OCR技术的智能图片重命名工具，支持中文姓名识别与自动命名，专为美术教育机构设计。

# 一画室智能图片命名 v4.5

## 项目简介

一画室智能图片命名 v4.5 是一款基于 OCR（光学字符识别）技术的自动图片重命名工具，适用于处理批量图片文件，并根据图片内容进行智能命名。本软件支持 EasyOCR 和 PaddleOCR 两种识别引擎，并提供多种优化策略，以提高识别准确率。

## 功能特点

- **多引擎支持**：支持 EasyOCR 和 PaddleOCR，可单独使用或组合使用。
- **智能匹配**：结合形近字映射、全名匹配、子串匹配、单字兜底等策略，提高识别准确率。
- **高效处理**：支持批量图片处理，并提供日志记录与进度显示。
- **可选图像预处理**：针对灰底白字等情况进行增强处理，提高识别效果。
- **灵活配置**：用户可自定义 OCR 识别模式、处理区域、预处理选项等。

## 运行环境

- **操作系统**：Windows 10/11
- **Python 版本**：3.12
- **依赖库**：
  - PyQt5
  - OpenCV (cv2)
  - EasyOCR
  - PaddleOCR
  - NumPy
  - Torch（如果使用 GPU 加速）

## 安装方法

### 方式 1：使用安装包（推荐）

直接运行提供的 Windows 安装包进行安装，无需手动配置环境。

### 方式 2：使用打包好的 exe 运行

无需安装 Python 及依赖库，直接运行 `main4.5.exe` 即可。

### 方式 3：手动安装运行（开发者模式）

1. 克隆项目：
   ```bash
   git clone https://github.com/yourusername/ImageRenamer.git
   cd ImageRenamer
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行程序：
   ```bash
   python main4.5.py
   ```

## 使用方法

1. **选择图片文件夹**：点击“选择图片文件夹”按钮，选定包含图片的目录。
2. **选择姓名库文件**：点击“选择姓名库文件”按钮，加载包含姓名的文本文件。
3. **配置 OCR 选项**（可选）：
   - 选择 OCR 引擎（EasyOCR、PaddleOCR、双重识别）。
   - 选择是否启用图像预处理、全图识别模式、单字兜底匹配。
4. **开始处理**：点击“开始处理”按钮，软件会自动识别图片中的姓名并重命名文件。
5. **查看日志与进度**：进度条显示当前处理进度，日志窗口记录详细处理信息。
6. **停止处理**（可选）：在任务执行过程中可随时点击“停止”按钮中断任务。

## 目录结构

```plaintext
main/
│── models/               # OCR 相关模型存储目录
│── main4.5.py            # 主程序
│── requirements.txt      # 依赖库列表
│── README.md             # 说明文档
│── LICENSE               # 许可证文件
└── dist/                 # 打包输出目录
```

## 依赖说明

软件在运行时需要以下 Python 库：

```bash
pip install PyQt5 opencv-python easyocr paddleocr numpy torch
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建你的特性分支 (`git checkout -b feature-branch`)
3. 提交你的修改 (`git commit -m 'Add some feature'`)
4. 推送到远程分支 (`git push origin feature-branch`)
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证，详情请参阅 [LICENSE](LICENSE) 文件。

## 联系方式

如有任何问题或建议，请联系 [[your-email@example.com](mailto\:your-email@example.com)] 或访问 GitHub 项目页面提交 Issue。

---

感谢使用“一画室智能图片命名 v4.5”！希望能帮助你提高工作效率 🎨📷✨

