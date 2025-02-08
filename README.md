# 一画室智能图片命名工具 v4.5

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-green)
![OCR](https://img.shields.io/badge/OCR-EasyOCR%20%26%20PaddleOCR-orange)

一款基于OCR技术的智能图片重命名工具，支持中文姓名识别与自动命名，专为美术教育机构设计。

## 功能特性 ✨

- **双引擎OCR识别**  
  ✔️ 支持 EasyOCR 和 PaddleOCR 引擎  
  ✔️ 可切换单引擎或双引擎协同工作模式

- **智能匹配算法**  
  ✔️ 四级匹配策略：形近字→全名→子串→单字兜底  
  ✔️ 内置相似字映射表（`similar_map.txt`）

- **图像处理优化**  
  ✔️ 自适应区域裁剪（默认右下角40%区域）  
  ✔️ 灰底白字专用预处理（对比度增强+高斯模糊）

- **批量处理能力**  
  ✔️ 支持 PNG/JPG/JPEG 格式批量处理  
  ✔️ 自动处理重名文件（追加序号）

- **可视化界面**  
  ✔️ 实时处理日志  
  ✔️ 进度条可视化  
  ✔️ 多语言支持（简体中文）

## 安装指南 🛠️

### 环境要求
- Python 3.12
- NVIDIA GPU（推荐，可加速OCR识别）

### 依赖安装
```bash
# 创建虚拟环境（推荐）
conda create -n ocr_rename python=3.12
conda activate ocr_rename

# 安装核心依赖
pip install PyQt5 opencv-python easyocr paddleocr numpy difflib2

使用说明 📖
快速启动：
python main4.5.py
操作流程
选择图片文件夹
➤ 包含待处理的图片文件（支持多层目录）

加载姓名库
➤ 文本文件格式（.txt），每行一个中文姓名
➤ 示例：

复制
张三
李四
王小明
设置识别参数
界面截图

🔘 OCR引擎选择（双引擎/单引擎）

✅ 图像预处理（推荐开启）

✅ 全图识别模式（根据需求选择）

开始处理
➤ 实时显示处理日志和进度
➤ 输出示例：

复制
14:23:05 - 正在处理: IMG_001.jpg (1/50)
14:23:07 - 重命名成功: IMG_001.jpg → 李四.jpg
高级配置 ⚙️
相似字映射表
编辑similar_map.txt文件添加自定义映射：

text
复制
"王": "王,玉,主"
"李": "李,季,子"
命令行参数
bash
复制
# 打包版本运行（需先使用PyInstaller打包）
./dist/main4.5.exe --gpu  # 强制启用GPU加速
常见问题 ❓
OCR识别不准确
确保图片清晰度（建议300dpi以上）

尝试启用「图像预处理」选项

检查姓名库是否包含目标姓名

GPU未启用
python
复制
# 在EnhancedOCR类中检查GPU状态
print(f"GPU可用状态: {self.gpu_available}")
处理速度慢
关闭不需要的OCR引擎

减少处理区域范围（关闭全图模式）

确保已启用GPU加速

贡献指南 🤝
欢迎提交PR！请遵循以下规范：

Fork本仓库

创建特性分支（feat/your-feature）

提交清晰的commit message

更新相关文档

许可证 📜
MIT License © 2023 一画室
