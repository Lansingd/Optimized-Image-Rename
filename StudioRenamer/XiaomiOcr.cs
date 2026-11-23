using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Text.Json;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows.Media.Imaging;

namespace StudioRenamer;

internal sealed record OcrReading(string Raw, string[] Names);

internal sealed class XiaomiOcr : IDisposable
{
    public const string Model = "longwayxu/xiaomi-ocr-0:bf16";
    public const int ParallelRequests = 4;
    private const string Prompt = """
        这是学生作品的照片。找出照片中真实可见的学生姓名或手写署名，逐字照抄。优先看“姓名”栏和作品边角的署名，也检查其他位置。不要把题目、日期、学校、画室或印刷标语当作姓名，不要猜测看不清的字。只输出 JSON，例如 {"names":["张明"]}。如果没有清晰署名输出 {"names":[]}。如果有多个可能的姓名，将它们都放进数组。
        """;
    private const string ReadCorner = "识别图片中的文字。";
    private const string CheckCorner = "请逐字转录图中最靠近底边的白色小字。只输出文字本身，无文字输出空。";
    private const string CertificatePrompt = """
        仔细读取证书上“姓名”栏内的汉字。如果看得清，逐字照抄姓名，不要用拼音推测，也不要把罕见字替换成常用字。只输出 {"name":"读到的姓名"}。看不清或没有姓名栏则输出 {"name":""}。
        """;
    private static readonly Regex CornerName = new(
        @"^[\p{IsCJKUnifiedIdeographs}\p{IsCJKUnifiedIdeographsExtensionA}]{2,4}$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private static readonly Regex LabeledName = new(
        @"^\s*(?:[-*#]\s*)?(?:\*\*)?(?:学生姓名|姓名水印|姓名|署名|落款|学生名字|获奖人|证书姓名)\s*[:：]\s*(?:\*\*)?\s*(?<name>[\p{L}·・.'’ \t]{2,48})\s*(?:\*\*)?\s*[。.]?\s*$",
        RegexOptions.Compiled | RegexOptions.Multiline | RegexOptions.CultureInvariant);
    private Process? process;
    private HttpClient? client;
    private volatile string startupError = "";

    public async Task StartAsync(CancellationToken token)
    {
        if (process is { HasExited: false } && client != null) return;
        Dispose();
        string runtime = Path.Combine(AppContext.BaseDirectory, "runtime");
        string binary = Path.Combine(runtime, "ollama", "ollama.exe");
        string models = Path.Combine(runtime, "models");
        if (!File.Exists(binary)) throw new FileNotFoundException("找不到本地 Ollama。请保留程序目录下的 runtime 文件夹。");
        if (!File.Exists(Path.Combine(models, "manifests", "registry.ollama.ai", "longwayxu", "xiaomi-ocr-0", "bf16")))
            throw new FileNotFoundException("找不到 Xiaomi-OCR-0 BF16 模型，请检查 runtime/models。");

        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        var start = new ProcessStartInfo(binary, "serve")
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(binary)!,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        start.Environment["OLLAMA_HOST"] = $"127.0.0.1:{port}";
        start.Environment["OLLAMA_MODELS"] = models;
        start.Environment["OLLAMA_NO_CLOUD"] = "1";
        start.Environment["OLLAMA_NOPRUNE"] = "1";
        start.Environment["OLLAMA_NUM_PARALLEL"] = ParallelRequests.ToString(System.Globalization.CultureInfo.InvariantCulture);
        start.Environment["OLLAMA_MAX_LOADED_MODELS"] = "1";
        startupError = "";
        try
        {
            process = Process.Start(start) ?? throw new InvalidOperationException("无法启动 Ollama。");
            process.ErrorDataReceived += (_, e) =>
            {
                if (e.Data is { } line && (line.Contains("error", StringComparison.OrdinalIgnoreCase) ||
                                            line.Contains("panic", StringComparison.OrdinalIgnoreCase)))
                    startupError = line;
            };
            process.BeginErrorReadLine();
            process.BeginOutputReadLine();
            var localClient = new HttpClient(new HttpClientHandler { UseProxy = false, AllowAutoRedirect = false })
            {
                BaseAddress = new Uri($"http://127.0.0.1:{port}"),
                Timeout = TimeSpan.FromMinutes(5)
            };
            client = localClient;
            for (int attempt = 0; attempt < 100; attempt++)
            {
                token.ThrowIfCancellationRequested();
                if (process.HasExited)
                    throw new IOException($"本地模型服务已退出。{startupError}");
                try
                {
                    using HttpResponseMessage response = await localClient.GetAsync("/api/version", token);
                    if (response.IsSuccessStatusCode) return;
                }
                catch (HttpRequestException) { }
                await Task.Delay(200, token);
            }

            throw new TimeoutException("本地模型服务启动超时。");
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public async Task<OcrReading> RecognizeAsync(BitmapSource image, bool lowerRight, CancellationToken token)
    {
        var (raw, truncated) = await GenerateAsync(PhotoImages.Prepare(image, lowerRight), Prompt, 128, token);
        return new OcrReading(raw, ReadNames(raw, truncated));
    }

    public async Task<OcrReading> RecognizeWatermarkAsync(BitmapSource corner, CancellationToken token)
    {
        byte[] png = PhotoImages.Prepare(corner, lowerRight: false);
        var (raw, truncated) = await GenerateAsync(png, ReadCorner, 64, token);
        if (truncated || !CornerName.IsMatch(raw)) return new OcrReading(raw, []);

        var (check, truncatedCheck) = await GenerateAsync(png, CheckCorner, 64, token);
        return new OcrReading($"{raw}\n复核：{check}", !truncatedCheck && raw == check ? [raw] : []);
    }

    public async Task<OcrReading> RecognizeCertificateAsync(BitmapSource region, CancellationToken token)
    {
        var (raw, truncated) = await GenerateAsync(PhotoImages.Prepare(region, lowerRight: false), CertificatePrompt, 96, token);
        return new OcrReading(raw, ReadNames(raw, truncated));
    }

    private async Task<(string Raw, bool Truncated)> GenerateAsync(byte[] png, string prompt, int tokens, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        // Single-image, non-thinking prompt and raw image marker specified by Xiaomi-OCR-0.
        string rendered = $"<|im_start|>user\n[img-0]{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        var payload = new
        {
            model = Model, prompt = rendered, images = new[] { Convert.ToBase64String(png) },
            raw = true, stream = false, truncate = false, keep_alive = "15m",
            options = new { temperature = 0, repeat_penalty = 1, seed = 0, num_ctx = 8192, num_predict = tokens }
        };

        using HttpResponseMessage response = await client!.PostAsJsonAsync("/api/generate", payload, token);
        if (!response.IsSuccessStatusCode)
        {
            string message = await response.Content.ReadAsStringAsync(token);
            throw new IOException($"模型请求失败（{(int)response.StatusCode}）{message[..Math.Min(message.Length, 350)]}");
        }

        using JsonDocument result = JsonDocument.Parse(await response.Content.ReadAsStreamAsync(token));
        JsonElement root = result.RootElement;
        string raw = root.GetProperty("response").GetString()?.Trim() ?? "";
        bool truncated = root.TryGetProperty("done_reason", out JsonElement reason) && reason.GetString() == "length";
        return (raw, truncated);
    }

    internal static string[] ReadNames(string raw, bool truncated = false)
    {
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        byte[] utf8 = Encoding.UTF8.GetBytes(raw);
        for (int offset = 0; offset < utf8.Length; offset++)
        {
            if (utf8[offset] != '{' && utf8[offset] != '[') continue;
            var reader = new Utf8JsonReader(utf8.AsSpan(offset), isFinalBlock: false, state: default);
            try
            {
                if (!JsonDocument.TryParseValue(ref reader, out JsonDocument? document)) continue;
                using (document)
                    CollectStructuredNames(document.RootElement, names);
                offset += (int)reader.BytesConsumed - 1;
            }
            catch (JsonException) { }
        }

        if (names.Count == 0 && !truncated)
        {
            foreach (Match entry in LabeledName.Matches(raw))
                AddName(entry.Groups["name"].Value, names);
        }
        return names.ToArray();
    }

    private static void CollectStructuredNames(JsonElement element, HashSet<string> names)
    {
        if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
                CollectStructuredNames(item, names);
            return;
        }
        if (element.ValueKind != JsonValueKind.Object) return;
        foreach (JsonProperty field in element.EnumerateObject())
        {
            if (field.Name.ToLowerInvariant() is not ("names" or "name" or "student_name" or "student_names" or "姓名" or "学生姓名" or "署名" or "姓名水印" or "获奖人" or "落款"))
                continue;
            if (field.Value.ValueKind == JsonValueKind.String)
                AddName(field.Value.GetString()!, names);
            else if (field.Value.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement name in field.Value.EnumerateArray())
                {
                    if (name.ValueKind == JsonValueKind.String) AddName(name.GetString()!, names);
                }
            }
        }
    }

    private static void AddName(string candidate, HashSet<string> names)
    {
        string name = candidate.Trim().Trim('「', '」', '“', '”', '"', '*', '\x60').Trim();
        if (name is not ("姓名" or "学生姓名" or "无" or "未知" or "看不清" or "无法识别" or "未识别" or "未找到" or "不清晰" or "不明确" or "None" or "null") &&
            name.Count(char.IsLetter) >= 2 && PhotoFiles.IsName(name))
            names.Add(name);
    }

    public void Dispose()
    {
        client?.Dispose();
        client = null;
        if (process == null) return;
        try
        {
            if (!process.HasExited) process.Kill(entireProcessTree: true);
        }
        catch (InvalidOperationException) { }
        finally
        {
            process.Dispose();
            process = null;
        }
    }
}
