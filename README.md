# 一画室智能图片命名工具
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-green)
![OCR](https://img.shields.io/badge/OCR-EasyOCR%20%26%20PaddleOCR-orange)

**一款基于OCR技术的智能图片重命名工具，支持中文姓名识别与自动命名，方便大批量图片的管理，专为美术教育机构设计。**

![image](https://github.com/Lansingd/Optimized-Image-Rename/blob/main/IMG/main.png)


## 项目简介

一画室智能图片命名 是一款基于 OCR（光学字符识别）技术的自动图片重命名工具，智能识别图片中的人名并重命名文件。
适用于处理批量图片文件，并根据图片内容进行智能命名。本软件支持 EasyOCR 和 PaddleOCR 两种识别引擎，并提供多种优化策略，以提高识别准确率。

## 效果展示

![image](https://github.com/Lansingd/Optimized-Image-Rename/blob/main/IMG/test/test1.png)
![image](https://github.com/Lansingd/Optimized-Image-Rename/blob/main/IMG/test/test2.png)

## 功能特点

- **多引擎支持**：支持 EasyOCR 和 PaddleOCR，可单独使用或组合使用。
- **智能匹配**：结合形近字映射、全名匹配、子串匹配、单字兜底等策略，提高识别准确率。
- **高效处理**：支持批量图片处理，并提供日志记录与进度显示。
- **可选图像预处理**：针对灰底白字等情况进行增强处理，提高识别效果。
- **灵活配置**：用户可自定义 OCR 识别模式、处理区域、预处理选项等。

## 运行环境

- **Python 版本**：3.11 3.12
- **CUDA 版本**：11.8（如果使用 GPU 加速）
- **CUDNN 版本**：8.9.7（如果使用 GPU 加速）
- **依赖库**：
  - PyQt5
  - OpenCV (cv2)
  - EasyOCR
  - PaddleOCR
  - NumPy
  - Torch（如果使用 GPU 加速）

## 运行方法

### 方式 1：使用安装包（不推荐）

直接运行提供的 Windows 安装包进行安装，无需手动配置环境。

### 方式 2：直接运行（推荐）

1. 克隆项目：
   ```bash
   git clone https://github.com/Lansingd/Optimized-Image-Rename.git
   cd Optimized-Image-Rename
   ```
2. 安装依赖（推荐在conda环境下）：
   ```bash
   python -m pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```
3. 运行程序：
   ```bash
   python main4.5.py
   ```

## 使用方法

1. **选择图片文件夹**：点击“选择图片文件夹”按钮，选定包含图片的目录。
2. **选择姓名库文件**：
➤ 文本文件格式（.txt），每行一个中文姓名
➤ 示例：
   ```
   张三
   李四
   王小明
   ```
3. **相似字映射表(可选)**：编辑similar_map.txt文件添加自定义映射：
➤ 示例：
   ```
   "孑": "子",
   "壬": "王",
   ```
5. **配置 OCR 选项**（可选）：
   - 选择 OCR 引擎（EasyOCR、PaddleOCR、双重识别）。
   - 选择是否启用图像预处理、全图识别模式、单字兜底匹配。
6. **开始处理**：点击“开始处理”按钮，软件会自动识别图片中的姓名并重命名文件。
7. **查看日志与进度**：进度条显示当前处理进度，日志窗口记录详细处理信息。
8. **停止处理**（可选）：在任务执行过程中可随时点击“停止”按钮中断任务。

## 目录结构

```plaintext
main/
│── models/               # OCR 相关模型存储目录
│── main4.5.py            # 主程序
│── similar_map.txt       # 相似字映射表
│── requirements.txt      # 依赖库列表
│── README.md             # 说明文档
│── LICENSE               # 许可证文件
└── dist/                 # 打包输出目录
```

## 更新日志
v1.0：实现基本图片重命名功能，使用 EasyOCR 进行 OCR 识别。
遍历指定文件夹中的图片文件，裁剪右下角区域进行 OCR 识别。
将识别的文本清理后作为新文件名，处理重名文件。

v1.1：改进图像预处理和姓名格式化，添加错误处理。
添加针对灰底白字的图像预处理（反色、对比度增强、模糊）。
规范姓名格式，只保留中文字符，优先取最后 3 个字。
添加错误处理和日志记录，记录处理失败的文件。

v2.0：引入姓名库和智能匹配，提高重命名准确性。
加载姓名库并构建姓名映射，支持智能匹配，优先匹配库中姓名。
添加模糊匹配标记，区分匹配可靠性。
优化 OCR 识别的预处理和参数。

v3.0：添加 GUI 界面和多线程处理，提升用户体验和效率。
添加 Tkinter GUI 界面，支持用户选择文件夹、姓名库文件和 OCR 引擎。
支持 EasyOCR 和 PaddleOCR 双引擎，以及双重识别模式。
添加图像预处理选项和全图识别模式。
实现多线程处理和进度条显示，改进日志记录和错误处理。


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

如有任何问题或建议，请联系 [[Lans3q@Gmail.com](mailto\:your-email@example.com)] 或访问 GitHub 项目页面提交 Issue。

---

感谢使用“一画室智能图片命名”！希望能帮助你提高工作效率 🎨📷✨

