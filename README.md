# 📸 一画室智能图片命名 - .NET 版 (v4.9) 🚀

![C#](https://img.shields.io/badge/-C%23-239120?style=flat&logo=c-sharp&logoColor=white) ![.NET](https://img.shields.io/badge/-.NET-512BD4?style=flat&logo=dotnet&logoColor=white) ![WPF](https://img.shields.io/badge/-WPF-0078D6?style=flat&logo=windows&logoColor=white) ![PaddleOCR](https://img.shields.io/badge/-PaddleOCR-FF6F00?style=flat&logo=paddlepaddle&logoColor=white) ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

🎉 **欢迎体验 一画室智能图片命名！** 这是一款专为我们一画室设计的智能图片批量重命名工具，也适用于广大美术教培机构，基于 `.NET` 和 `WPF`，结合强大的 **PaddleOCR** 中文识别技术，轻松从图片中提取中文姓名并智能整理！ 📂✨

---

## 🌟 为什么制作 一画室智能图片命名？
      因为每学期整理学生图片实在是太累了……

🔍 **从 Python 到 .NET 的华丽转身**  
相比之前基于 `PyQt5` 的 Python 版本（`v4.7`），我将工具移植到 `.NET / WPF`，带来更流畅的用户体验和更便捷的部署方式！告别繁琐的依赖配置，享受开箱即用的快感！ 🎯

### 🛠️ Python 版本的痛点
- 🛑 依赖复杂：PyTorch、EasyOCR、PaddleOCR 需手动配置，CUDA 环境更是噩梦！
- 📦 打包体积巨大，动辄数GB，部署困难。
- 😓 非技术用户启动门槛高，难以普及。

### 🚀 .NET 版本的优势
- ✅ **双击运行**：提供 `.exe` 或 `MSI` 安装包，Windows 用户即刻上手！
- 🖼️ **原生界面**：基于 WPF 的现代化 UI，符合 Windows 操作习惯。
- 📥 **依赖管理**：通过 NuGet 自动管理依赖，模型缓存到 `%AppData%`，安装无忧。
- 🧹 **内存优化**：通过 `using` 和 `IDisposable` 严格控制 OpenCV 和 PaddleOCR 资源，稳定可靠。
- 📏 **轻量分发**：告别 Python 打包的庞大体积，部署更轻松！

### ⚖️ 不足与取舍
- 🐢 当前版本仅支持 **CPU 推理**，速度稍逊于 Python + GPU 方案，得益于C#的效率优势，实际差距不大。
- 🔄 并发处理受限，为避免 PaddleOCR 锁死，采用串行锁策略。
- ⚙️ GPU 部署仍推荐 Python 环境，但配置复杂，适合技术用户。

---

## 🎨 核心功能

- 🀄 **PaddleOCR 中文识别**：集成 PP-OCRv4 模型，精准提取图片中的中文姓名，模型自动缓存到 `%AppData%`。
- 🖼️ **OpenCV 图像预处理**：轻量优化图像，提升 OCR 识别稳定性。
- 📍 **多 ROI 检测**：优先识别右下角姓名区域，全图兜底，完美适配美术教育场景。
- 🧠 **智能姓名清洗**：自动过滤不符合姓名特征的文本，输出准确结果。
- 🔒 **串行执行**：全程串行处理，杜绝 PaddleOCR 并发调用时的锁死问题。
- 📂 **智能文件夹管理**：根据识别的人名自动创建文件夹，图片自动归档，整理更高效！

---

## 🎉 v4.9 更新亮点

1. 🀄 **形近字匹配优化**  
   - 修复了 `similar_map.txt` 未生效的问题，OCR 识别的中文现可智能替换为更合理的人名，避免“鬼画符”输出。
2. 📂 **智能文件夹管理**  
   - 自动根据识别的人名创建对应文件夹，图片一键归档，批量整理从未如此简单！
3. 🍗 **开发者小记**  
   - *今晚吃了德四家的烧鸡，香到飞起！😋🔥*

---

## 🏫 适用场景

- 🖌️ **教育机构**：快速整理学生作业图片，按图片上的姓名自动重命名并归档。
- 📚 **档案管理**：批量处理扫描文档，提取中文姓名并规范命名。
- 🖼️ **美术教育**：针对手写签名图片，精准识别姓名并整理文件。

---

## 📦 安装与使用

1. **下载安装包**  
   从 [Releases](https://github.com/Lansingd/Optimized-Image-Rename/releases) 下载最新 `.exe` 或 `.MSI` 文件。
2. **运行程序**  
   双击运行，Windows 自动完成依赖安装，模型会缓存到 `%AppData%`。
3. **选择图片文件夹**  
   在 WPF 界面中选择需要处理的图片文件夹和学生姓名库，点击“开始”即可自动识别并重命名！

---

## 📜 开源许可

![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)  
本项目采用 **MIT 许可**，欢迎 Fork、Star 和贡献代码！🌟

---

## 🙌 贡献与反馈

- 🐛 发现 Bug？请在 [Issues](https://github.com/Lansingd/Optimized-Image-Rename/issues) 提交问题。
- 💡 有新想法？欢迎提交 Pull Request 或联系我们！
- ⭐ 喜欢这个项目？点个 Star 支持一下吧！😄

---

## 📢 关于

**Optimized Image Renamer** 是一款专为中文环境设计的智能图片重命名工具，结合 PaddleOCR 和 OpenCV 技术，助力教育与档案管理场景高效处理图片文件。  
💌 欢迎加入我们的开源社区，共同打造更好用的工具！

© 2025 Lansingd, proudly powered by .NET and WPF.
