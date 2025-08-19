📖 分支说明：C# / WPF 版 ImageRenamer

这个分支的目标，是把原本的 Python + PaddleOCR 工具移植到 C# / WPF 桌面应用。

这样做的主要原因：

Python 版本在 Windows 下依赖繁琐，部署困难；

希望提供一个双击即可运行的 Windows 桌面程序，降低使用门槛；

结合 WPF UI，可以提供更友好的操作体验（批量重命名、日志输出、参数选择等）。

✅ 好处 / 改进点
免安装 Python 环境：用户不再需要配置 pip / venv / PaddleOCR 依赖。

一键打包成安装包 (MSI/EXE)：支持 Windows 下直接安装，图标、快捷方式等体验接近普通软件。

UI 界面操作：相比命令行，提供更直观的复选框、日志窗口、进度反馈。

跨显卡兼容：在 .NET 里可以选择

PaddleOCR (准确率较高)

或 ONNX Runtime + DirectML（支持 NVIDIA / AMD / Intel GPU，无需用户安装 CUDA）。

配置自动记忆：如“轻量预处理”选项会记住上次选择。

⚠️ 不足 / 取舍
识别准确率：目前 PaddleOCR C# 封装版本的精度 ≈ Python 原版，但 DirectML + ONNX 的识别率仍偏低；推荐默认使用 PaddleOCR 模式。

性能：在 CPU 下比 Python 稍慢；GPU 下因 DirectML 抽象层，速度比 CUDA 版本慢一些。

生态差异：Python 社区对 OCR 的模型支持更快（如 PP-OCRv5/PP-Structure 等），C# 分支可能会滞后。

模型下载：有时依赖 PaddleOCR 模型自动下载，受网络影响可能失败，需要用户手动放置。

📦 适用场景
想在 Windows 下快速体验 OCR + 批量重命名，而不想折腾 Python 环境；

需要一个带 UI 的轻量工具，方便非技术用户使用；

公司/团队环境，直接分发 MSI/EXE 即可部署，无需安装解释器。
