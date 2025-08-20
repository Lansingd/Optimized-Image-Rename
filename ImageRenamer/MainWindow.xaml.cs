using Microsoft.Win32;
using Ookii.Dialogs.Wpf;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

namespace ImageRenamer;

public partial class MainWindow : Window
{
    private string? _imageFolder;
    private string? _nameFile;
    private string? _destRoot; // 目标根目录（可选）
    private CancellationTokenSource? _cts;

    private Services.OcrService? _ocr;
    private Services.NameMatcher? _matcher;
    private Services.SimilarMap _similar = new();

    public MainWindow()
    {
        InitializeComponent();
        Log("欢迎使用一画室图片命名器");
    }

    private void OnPickFolder(object sender, RoutedEventArgs e)
    {
        var dlg = new VistaFolderBrowserDialog
        {
            Description = "选择图片文件夹"
        };
        var ok = dlg.ShowDialog(this);
        if (ok == true)
        {
            _imageFolder = dlg.SelectedPath;
            TxtFolder.Text = _imageFolder;
            Log($"已选择图片文件夹: {_imageFolder}");
        }
    }

    private void OnPickDestRoot(object sender, RoutedEventArgs e)
    {
        var dlg = new Ookii.Dialogs.Wpf.VistaFolderBrowserDialog
        {
            Description = "选择自动建文件夹的根目录（可选）"
        };
        var ok = dlg.ShowDialog(this);
        if (ok == true)
        {
            _destRoot = dlg.SelectedPath;
            TxtDestRoot.Text = _destRoot;
            Log($"已选择目标根目录: {_destRoot}");
        }
    }


    private void OnPickNameFile(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Filter = "文本文件 (*.txt)|*.txt|所有文件 (*.*)|*.*"
        };
        if (dlg.ShowDialog() == true)
        {
            _nameFile = dlg.FileName;
            TxtNameFile.Text = _nameFile;
            Log($"已选择姓名库文件: {_nameFile}");
        }
    }

    private async void OnStart(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(_imageFolder) || string.IsNullOrWhiteSpace(_nameFile))
        {
            MessageBox.Show("请先选择图片文件夹和姓名库文件", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        BtnStart.IsEnabled = false;
        _cts = new CancellationTokenSource();

        try
        {
            // 1) 加载 similar_map（可选）
            _similar.TryLoadFromFile(Path.Combine(AppContext.BaseDirectory, "similar_map.txt"));
            if (_similar.Count > 0) Log($"已加载 {_similar.Count} 条形近字映射");

            // 2) 加载姓名库
            var names = await Services.NameMatcher.LoadNamesAsync(_nameFile);
            Log($"成功加载 {names.Count} 个姓名");
            _matcher = new Services.NameMatcher(names, _similar, ChkSingleFallback.IsChecked == true);

            // 3) 确保 chi_sim 模型可用

            // 4) 初始化 OCR
            _ocr = await Services.OcrService.CreateAsync(
               preprocess: ChkPreprocess.IsChecked == true,
               log: Log,
               ct: _cts.Token);

            // 5) 扫描图像并处理
            var files = Directory.EnumerateFiles(_imageFolder!, "*.*")
                                 .Where(f => f.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
                                          || f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)
                                          || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase))
                                 .ToList();

            Bar.Minimum = 0; Bar.Maximum = files.Count; Bar.Value = 0;

            Log($"开始处理，共 {files.Count} 张图片…");
            int idx = 0;

            foreach (var file in files)
            {
                _cts.Token.ThrowIfCancellationRequested();
                Log($"正在处理: {Path.GetFileName(file)} ({idx + 1}/{files.Count})");

                try
                {
                    // 识别（全图/右下角区域）
                    var text = await _ocr.RecognizeAsync(file, areaFull: ChkFull.IsChecked == true, _cts.Token);
                    Log($"Tesseract 结果: {text}");

                    // 匹配
                    // 匹配
                    var best = _matcher!.FindBest(text);
                    if (best != null)
                    {
                        var ext = Path.GetExtension(file);
                        var safe = Services.NameMatcher.ToSafeFilename(best);

                        // 1) 计算目标基路径
                        var autoFolder = ChkAutoFolder.IsChecked == true;
                        var targetBase = autoFolder
                            ? (_destRoot ?? _imageFolder!)   // 勾选但未选根目录 → 默认用图片文件夹
                            : _imageFolder!;                 // 未勾选 → 保持原逻辑

                        // 2) 如需要，创建按姓名分组的子目录
                        var targetDir = autoFolder
                            ? Path.Combine(targetBase, safe) // 每个姓名一个同名文件夹
                            : targetBase;

                        try
                        {
                            if (!Directory.Exists(targetDir))
                                Directory.CreateDirectory(targetDir);
                        }
                        catch (Exception ex)
                        {
                            Log($"创建文件夹失败 [{targetDir}]: {ex.Message}，将回退到直接重命名。");
                            targetDir = _imageFolder!; // 回退策略：直接放回原图目录
                        }

                        // 3) 生成去重后的目标文件名
                        var newName = Path.Combine(targetDir, $"{safe}{ext}");
                        int c = 1;
                        while (File.Exists(newName))
                        {
                            newName = Path.Combine(targetDir, $"{safe}_{c}{ext}");
                            c++;
                        }

                        // 4) 移动（含重命名）
                        File.Move(file, newName);

                        // 5) 友好日志
                        if (autoFolder)
                            Log($"重命名并归档: {Path.GetFileName(file)} → {safe}/{Path.GetFileName(newName)}");
                        else
                            Log($"重命名成功: {Path.GetFileName(file)} → {Path.GetFileName(newName)}");
                    }

                }
                catch (OperationCanceledException) { throw; }
                catch (Exception ex)
                {
                    Log($"处理失败 [{Path.GetFileName(file)}]: {ex.Message}");
                }
                finally
                {
                    Bar.Value = ++idx;
                }
            }

            Log("处理完成！");
        }
        catch (OperationCanceledException)
        {
            Log("处理已中止");
        }
        catch (Exception ex)
        {
            Log("严重错误: " + ex);
        }
        finally
        {
            BtnStart.IsEnabled = true;
            _cts = null;
            _ocr?.Dispose();
        }
    }

    private void OnStop(object sender, RoutedEventArgs e)
    {
        _cts?.Cancel();
        Log("正在停止当前任务…");
    }

    private void Log(string msg)
    {
        Dispatcher.Invoke(() =>
        {
            TxtLog.AppendText($"{DateTime.Now:HH:mm:ss} [OCR] - {msg}{Environment.NewLine}");
            TxtLog.ScrollToEnd();
        });
    }
}
