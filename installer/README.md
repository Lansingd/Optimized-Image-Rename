# 构建 5.0 Windows 安装器

构建环境需要 .NET 8 SDK、Inno Setup 6.7 或兼容版本，以及相应的简体中文语言文件。简体中文文件可从 Inno Setup 源码仓库的 Files/Languages/ChineseSimplified.isl 获取。

在仓库根目录使用 PowerShell 运行下方命令，将路径替换为本机实际位置。

    powershell -ExecutionPolicy Bypass -File .\installer\build.ps1 -Compiler "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" -ChineseMessages "C:\tools\ChineseSimplified.isl"

安装包默认生成至 installer/artifacts。构建脚本发布自包含的 win-x64 桌面程序。Setup.iss 固定使用经过校验的 Ollama Windows 运行时和 Xiaomi-OCR-0 BF16 模型文件。安装器联网下载这些文件，并在安装过程中校验 SHA-256。修改模型版本时，需同步更新 model-manifest.json、下载地址、文件大小和哈希。
