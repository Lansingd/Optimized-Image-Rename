using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using Microsoft.Win32;

namespace StudioRenamer;

public partial class MainWindow : Window
{
    private readonly ObservableCollection<Photo> photos = [];
    private readonly XiaomiOcr ocr = new();
    private readonly Dictionary<Photo, OcrReading> readings = [];
    private HashSet<string> roster = new(StringComparer.OrdinalIgnoreCase);
    private string? sourceFolder;
    private string? outputFolder;
    private CancellationTokenSource? operation;
    private int previewRequest;
    private BitmapSource? previewImage;

    public MainWindow()
    {
        InitializeComponent();
        PhotoList.ItemsSource = photos;
    }

    private async void PickSource(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog { Title = "选择装有学生作品照片的目录", InitialDirectory = sourceFolder ?? "" };
        if (dialog.ShowDialog(this) != true) return;
        if (photos.Any(IsReady) && MessageBox.Show(this, "切换目录会放弃当前待整理的图片。现在切换吗？",
            Title, MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes) return;
        Busy(true, indeterminate: true);
        Message.Text = "正在读取图片目录…";
        try
        {
            Photo[] found = await Task.Run(() => PhotoFiles.Scan(dialog.FolderName));
            sourceFolder = dialog.FolderName;
            SourcePath.Text = sourceFolder;
            outputFolder = null;
            UpdateOutputLabel();
            foreach (Photo photo in photos) photo.PropertyChanged -= UpdateButtons;
            photos.Clear();
            readings.Clear();
            foreach (Photo photo in found)
            {
                photo.PropertyChanged += UpdateButtons;
                photos.Add(photo);
            }
            ListCount.Text = $"{found.Length} 张图片";
            EmptyList.Text = "这个目录没有可识别的图片";
            EmptyList.Visibility = found.Length == 0 ? Visibility.Visible : Visibility.Collapsed;
            PhotoList.SelectedIndex = found.Length == 0 ? -1 : 0;
            Message.Text = found.Length == 0 ? "请选择包含图片的目录" : "点击「开始识别」，或直接在右侧填写姓名";
        }
        catch (Exception ex) { Message.Text = $"无法读取目录。{ex.Message}"; }
        finally { Busy(false); }
    }

    private void PickRoster(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog { Title = "导入姓名名单", Filter = "姓名名单 (*.txt;*.csv)|*.txt;*.csv|所有文件|*.*" };
        if (dialog.ShowDialog(this) != true) return;
        try
        {
            var names = PhotoFiles.ReadRoster(dialog.FileName);
            if (names.Count == 0)
            {
                Message.Text = "名单中没有读到姓名。请使用单列名单，每行一个姓名。";
                return;
            }
            roster = names;
            RosterPath.Text = $"{roster.Count} 人 · {Path.GetFileName(dialog.FileName)}";
            ClearRosterButton.Visibility = Visibility.Visible;
            RefreshRosterMatches();
            Message.Text = $"已导入 {roster.Count} 个姓名";
        }
        catch (Exception ex) { Message.Text = $"无法读取名单。{ex.Message}"; }
    }

    private void ClearRoster(object sender, RoutedEventArgs e)
    {
        roster.Clear();
        RosterPath.Text = "可选，支持 TXT / CSV";
        ClearRosterButton.Visibility = Visibility.Collapsed;
        RefreshRosterMatches();
        Message.Text = "已清除名单";
    }

    private void RefreshRosterMatches()
    {
        foreach (var (photo, reading) in readings)
        {
            if (photo.Output == null && photo.Status != "已编辑" && photo.Status != "写入失败")
                ShowReading(photo, reading);
        }
        RefreshRosterSuggestions();
    }

    private void PickOutput(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog { Title = "选择姓名归档目录", InitialDirectory = outputFolder ?? sourceFolder ?? "" };
        if (dialog.ShowDialog(this) != true) return;
        outputFolder = dialog.FolderName;
        UpdateOutputLabel();
    }

    private void OutputModeChanged(object sender, SelectionChangedEventArgs e)
    {
        if (OutputPath != null) UpdateOutputLabel();
    }

    private void UpdateOutputLabel()
    {
        bool rename = OutputMode.SelectedIndex == 1;
        PickOutputButton.Visibility = rename ? Visibility.Collapsed : Visibility.Visible;
        OutputPath.Text = rename ? sourceFolder ?? "使用图片原目录" :
            outputFolder ?? (sourceFolder == null ? "默认使用图片目录中的「已整理」" : Path.Combine(sourceFolder, "已整理"));
    }

    private async void OpenPhoto(object sender, SelectionChangedEventArgs e)
    {
        if (Preview == null) return;
        var photo = PhotoList.SelectedItem as Photo;
        int request = ++previewRequest;
        PhotoEditor.DataContext = photo;
        RefreshRosterSuggestions();
        previewImage = null;
        Preview.Source = null;
        ZoomCorner.IsEnabled = false;
        PreviewStatus.Text = photo == null ? "选择一张图片查看" : "正在加载预览…";
        UpdateButtons(null, null);
        if (photo == null) return;
        try
        {
            var image = await Task.Run(() => PhotoImages.Read(photo.Output ?? photo.Source, 1080));
            if (request != previewRequest) return;
            previewImage = image;
            ZoomCorner.IsEnabled = true;
            ToggleZoom(this, new RoutedEventArgs());
            PreviewStatus.Text = "";
        }
        catch (Exception ex)
        {
            if (request == previewRequest) PreviewStatus.Text = $"无法打开预览。{ex.Message}";
        }
    }

    private void ToggleZoom(object sender, RoutedEventArgs e)
    {
        if (previewImage != null)
            Preview.Source = ZoomCorner.IsChecked == true ? PhotoImages.LowerRight(previewImage) : previewImage;
    }

    private async void StartScan(object sender, RoutedEventArgs e)
    {
        Photo[] pending = photos.Where(photo => photo.NeedsScan).ToArray();
        if (pending.Length == 0) return;
        using var cancel = new CancellationTokenSource();
        operation = cancel;
        Busy(true, indeterminate: true);
        Message.Text = "正在启动本地模型…";
        int done = 0, errors = 0;
        try
        {
            await ocr.StartAsync(cancel.Token);
            Progress.IsIndeterminate = false;
            Progress.Maximum = pending.Length;
            foreach (Photo photo in pending)
            {
                cancel.Token.ThrowIfCancellationRequested();
                Message.Text = $"正在识别 {done + 1}/{pending.Length} · {photo.FileName}";
                try
                {
                    OcrReading reading = await Task.Run(() => ReadPhoto(photo.Source, cancel.Token), cancel.Token);
                    cancel.Token.ThrowIfCancellationRequested();
                    readings[photo] = reading;
                    ShowReading(photo, reading);
                }
                catch (OperationCanceledException) when (cancel.IsCancellationRequested) { throw; }
                catch (Exception ex)
                {
                    errors++;
                    photo.Scanned("", "", false, "识别失败", ex.Message, retry: true);
                }
                Progress.Value = ++done;
            }
            Message.Text = $"已检查 {done} 张 · {errors} 张失败 · {photos.Count(IsReady)} 张已勾选";
        }
        catch (OperationCanceledException) when (cancel.IsCancellationRequested)
        {
            Message.Text = $"已停止 · 本次完成 {done}/{pending.Length} 张";
        }
        catch (Exception ex) { Message.Text = ex.Message; }
        finally
        {
            operation = null;
            Busy(false);
        }
    }

    private async Task<OcrReading> ReadPhoto(string path, CancellationToken token)
    {
        var corner = PhotoImages.Read(path, 1280, watermark: true);
        var watermark = await ocr.RecognizeWatermarkAsync(corner, token);
        if (watermark.Names.Length == 1 && (roster.Count == 0 || roster.Contains(watermark.Names[0])))
            return watermark with { Raw = $"右下角水印\n{watermark.Raw}" };

        var image = PhotoImages.Read(path, 2048);
        var full = await ocr.RecognizeAsync(image, lowerRight: false, token);
        string raw = $"右下角水印\n{watermark.Raw}\n\n全图\n{full.Raw}";
        var candidates = watermark.Names.Concat(full.Names);
        if (full.Raw.Contains("证书", StringComparison.Ordinal) || full.Raw.Contains("证件类型", StringComparison.Ordinal))
        {
            var certificate = await ocr.RecognizeCertificateAsync(PhotoImages.ReadCertificateRegion(path), token);
            raw += $"\n\n证书姓名栏\n{certificate.Raw}";
            candidates = candidates.Concat(certificate.Names);
        }
        var names = candidates.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        if (names.Length > 0) return new OcrReading(raw, names);

        var signature = await ocr.RecognizeAsync(image, lowerRight: true, token);
        return signature with { Raw = $"{raw}\n\n右下角落款\n{signature.Raw}" };
    }

    private void ShowReading(Photo photo, OcrReading reading)
    {
        if (reading.Names.Length == 0)
            photo.Scanned(reading.Raw, "", false, "待确认", "未获得明确姓名，可查看图片后手动填写。");
        else if (reading.Names.Length > 1)
            photo.Scanned(reading.Raw, "", false, "多个候选", $"请从照片中确认姓名。候选为 {string.Join("、", reading.Names)}。");
        else
        {
            string name = reading.Names[0];
            if (roster.Count == 0)
                photo.Scanned(reading.Raw, name, true, "已识别", "请核对照片上的姓名。");
            else if (roster.TryGetValue(name, out string? match))
                photo.Scanned(reading.Raw, match, true, "名单匹配", "识别结果与名单一致。");
            else
                photo.Scanned(reading.Raw, name, false, "名单外", "请修正姓名，或核实后勾选处理。");
        }
        if (ReferenceEquals(PhotoList.SelectedItem, photo)) RefreshRosterSuggestions();
    }

    private void RefreshRosterSuggestions()
    {
        string[] suggestions = PhotoList.SelectedItem is Photo { Output: null } photo &&
            readings.TryGetValue(photo, out var reading)
            ? PhotoFiles.RosterSuggestions(reading.Names, roster)
            : [];
        RosterSuggestionList.ItemsSource = suggestions;
        RosterSuggestionPanel.Visibility = suggestions.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    private void ChooseRosterName(object sender, RoutedEventArgs e)
    {
        if (!Settings.IsEnabled || PhotoList.SelectedItem is not Photo { Output: null } photo ||
            sender is not Button { Content: string name })
            return;

        photo.Name = name;
        photo.Included = true;
    }

    private async void Save(object sender, RoutedEventArgs e)
    {
        var batch = photos.Where(IsReady).Select(photo => (Photo: photo, Name: photo.Name.Trim())).ToArray();
        if (batch.Length == 0 || sourceFolder == null) return;
        bool rename = OutputMode.SelectedIndex == 1;
        string? archive = rename ? null : outputFolder ?? Path.Combine(sourceFolder, "已整理");
        if (rename && MessageBox.Show(this, $"将在原目录重命名 {batch.Length} 张已勾选图片。是否继续？",
            Title, MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes) return;

        Busy(true, indeterminate: false);
        Progress.Maximum = batch.Length;
        int written = 0, failed = 0;
        try
        {
            foreach (var (photo, name) in batch)
            {
                Message.Text = $"正在整理 {written + failed + 1}/{batch.Length} · {photo.FileName}";
                try
                {
                    string path = await Task.Run(() => PhotoFiles.Write(photo.Source, name, archive));
                    photo.Saved(path, rename ? "已重命名" : "已归档");
                    if (ReferenceEquals(PhotoList.SelectedItem, photo)) RefreshRosterSuggestions();
                    written++;
                }
                catch (Exception ex)
                {
                    photo.SaveFailed(ex.Message);
                    failed++;
                }
                Progress.Value = written + failed;
            }
            Message.Text = failed == 0 ? $"已整理 {written} 张 · {archive ?? sourceFolder}" : $"已整理 {written} 张 · {failed} 张失败，点选对应行查看原因";
        }
        finally { Busy(false); }
    }

    private static bool IsReady(Photo photo) => photo.Included && photo.Output == null && PhotoFiles.IsName(photo.Name.Trim());

    private void Stop(object sender, RoutedEventArgs e)
    {
        StopButton.IsEnabled = false;
        operation?.Cancel();
    }

    private void Busy(bool busy, bool indeterminate = false)
    {
        Settings.IsEnabled = !busy;
        PhotoList.IsHitTestVisible = !busy;
        PhotoList.IsEnabled = !busy;
        Progress.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;
        Progress.IsIndeterminate = indeterminate;
        Progress.Value = 0;
        StopButton.Visibility = operation != null ? Visibility.Visible : Visibility.Collapsed;
        StopButton.IsEnabled = true;
        UpdateButtons(null, null);
    }

    private void UpdateButtons(object? sender, PropertyChangedEventArgs? e)
    {
        ScanButton.IsEnabled = Settings.IsEnabled && photos.Any(photo => photo.NeedsScan);
        SaveButton.IsEnabled = Settings.IsEnabled && photos.Any(IsReady);
        NameEditor.IsEnabled = Settings.IsEnabled && PhotoList.SelectedItem is Photo { Output: null };
        RosterSuggestionPanel.IsEnabled = Settings.IsEnabled;
    }

    protected override void OnClosing(CancelEventArgs e)
    {
        if (!Settings.IsEnabled)
        {
            operation?.Cancel();
            e.Cancel = true;
            return;
        }
        ocr.Dispose();
        base.OnClosing(e);
    }
}
