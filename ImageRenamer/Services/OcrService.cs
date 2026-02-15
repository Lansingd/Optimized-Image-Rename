using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
// 新增：
using System.Text.RegularExpressions;
using OpenCvSharp;
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models;
using Sdcb.PaddleOCR.Models.Online;

namespace ImageRenamer.Services
{
    /// <summary>
    /// PaddleOCR（CPU）+ OpenCV 全流程版本：
    /// - 串行执行避免并发导致的原生库死锁
    /// - 全程使用 OpenCV Mat，严格 using 释放，杜绝 GDI+ 瓶颈
    /// - 多 ROI（右下角优先）+ 轻量预处理（OpenCV）
    /// </summary>
    public sealed class OcrService : IDisposable
    {
        private readonly bool _preprocess;
        private readonly PaddleOcrAll _ocr;

        // 新增：形近字映射
        private readonly List<KeyValuePair<string, string>> _similarPairs;

        private static readonly SemaphoreSlim _serial = new(1, 1);

        private OcrService(FullOcrModel model, bool preprocess)
        {
            _preprocess = preprocess;

            _ocr = new PaddleOcrAll(model, cfg =>
            {
                cfg.UseGpu = false;
            })
            {
                AllowRotateDetection = true,
                Enable180Classification = true,
            };

            // 新增：加载 similar_map.txt（默认在程序目录）
            var mapPath = Path.Combine(AppContext.BaseDirectory, "similar_map.txt");
            _similarPairs = LoadSimilarMap(mapPath);
        }


        public static async Task<OcrService> CreateAsync(bool preprocess, Action<string>? log = null, CancellationToken ct = default)
        {
            log?.Invoke("正在准备 PaddleOCR 中文模型（PP-OCRv4）…");
            var model = await OnlineFullModels.ChineseV4.DownloadAsync(ct); // 自动缓存到 %AppData%\paddleocr-models
            log?.Invoke("模型就绪。");

            var svc = new OcrService(model, preprocess);

            // 预热：避免首张图片初始化开销过大导致“像卡死”
            await Task.Run(() =>
            {
                using var warm = new Mat(64, 64, MatType.CV_8UC3, Scalar.White);
                try { _ = svc._ocr.Run(warm); } catch { /* 忽略预热异常 */ }
            }, ct);

            return svc;
        }

        /// <summary>
        /// 识别单张图片，返回清洗后的文本
        /// </summary>
        public async Task<string> RecognizeAsync(string imagePath, bool areaFull, CancellationToken ct)
        {
            await _serial.WaitAsync(ct); // 串行化，避免原生库并发问题
            try
            {
                return await Task.Run(() =>
                {
                    ct.ThrowIfCancellationRequested();

                    // 直接用 OpenCV 读图（避免 Image.FromFile 占用/锁文件/解码两次）
                    using var src = Cv2.ImRead(imagePath, ImreadModes.Color);
                    if (src.Empty()) return string.Empty;

                    var rois = areaFull
                        ? new List<Rect> { new Rect(0, 0, src.Width, src.Height) }
                        : GenerateRois(src.Width, src.Height);

                    string best = string.Empty;
                    int bestCjk = 0;

                    foreach (var r in rois)
                    {
                        // 边界裁剪
                        var rr = ClampRect(r, src.Width, src.Height);
                        if (rr.Width <= 0 || rr.Height <= 0) continue;

                        using var roi = new Mat(src, rr);
                        using var crop = roi.Clone(); // SubMat 需 Clone 后再处理/传入

                        using var proc = _preprocess ? PreprocessCv(crop) : crop.Clone();

                        // 有些图片会 180°，交给内置分类 + 旋转；无需我们手动转
                        PaddleOcrResult result;
                        try
                        {
                            result = _ocr.Run(proc); // PaddleOcrResult是托管对象，不需要手动释放原生资源
                        }
                        catch
                        {
                            // 某些极端图片可能触发底层异常，继续下一个 ROI
                            continue;
                        }

                        string text = string.Concat(result.Regions?.Select(z => z.Text) ?? Enumerable.Empty<string>());
                        text = Normalize(text);
                        text = ApplySimilarMap(text, _similarPairs);

                        int cjk = text.Count(ch => ch >= 0x4e00 && ch <= 0x9fff);
                        if (cjk == 0) continue;

                        if ((cjk >= 2 && cjk <= 4) || cjk > bestCjk)
                        {
                            best = text; bestCjk = cjk;
                            if (cjk >= 2 && cjk <= 4) break; // 早停：像姓名就定了
                        }
                    }

                    return best;
                }, ct);
            }
            finally
            {
                _serial.Release();
            }
        }

        // 新增：形近字映射加载与应用
        private static List<KeyValuePair<string, string>> LoadSimilarMap(string path)
        {
            var pairs = new List<KeyValuePair<string, string>>();
            try
            {
                if (!File.Exists(path)) return pairs;

                foreach (var raw in File.ReadAllLines(path, Encoding.UTF8))
                {
                    var s = raw.Trim();
                    if (string.IsNullOrEmpty(s)) continue;
                    if (s.StartsWith("#") || s.StartsWith("//")) continue; // 支持注释
                    if (s.EndsWith(",")) s = s[..^1]; // 容忍行尾逗号

                    // 解析形如  "错":"正"
                    var m = Regex.Match(s, "^\\s*\"(?<k>.*?)\"\\s*:\\s*\"(?<v>.*?)\"\\s*$");
                    if (m.Success)
                    {
                        var k = m.Groups["k"].Value;
                        var v = m.Groups["v"].Value;
                        if (!string.IsNullOrEmpty(k))
                            pairs.Add(new KeyValuePair<string, string>(k, v));
                        continue;
                    }

                    // 兜底：支持  错->正 / 错 正 / 错\t正  等简写
                    var parts = s.Split(new[] { "->", "=>", "\t", " " }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2)
                    {
                        var k = parts[0].Trim('"');
                        var v = parts[1].Trim('"');
                        if (!string.IsNullOrEmpty(k))
                            pairs.Add(new KeyValuePair<string, string>(k, v));
                    }
                }

                // 为避免短 key 抢先替换造成“误吞”，按 key 长度降序
                pairs = pairs
                    .OrderByDescending(p => p.Key.Length)
                    .ToList();
            }
            catch
            {
                // 读取/解析失败时静默，不影响 OCR 主流程
            }
            return pairs;
        }

        private static string ApplySimilarMap(string s, List<KeyValuePair<string, string>> pairs)
        {
            if (string.IsNullOrEmpty(s) || pairs.Count == 0) return s;

            // 做 1~2 轮以覆盖连锁替换，但避免死循环
            var prev = s;
            for (int round = 0; round < 2; round++)
            {
                foreach (var kv in pairs)
                {
                    if (string.IsNullOrEmpty(kv.Key)) continue;
                    s = s.Replace(kv.Key, kv.Value);
                }
                if (s == prev) break;
                prev = s;
            }
            return s;
        }


        // --------- ROI：右下角优先 + 边条 + 全图 ----------
        private static List<Rect> GenerateRois(int w, int h)
        {
            var L = new List<Rect>();
            Rect R(double x, double y, double rw, double rh)
            {
                int X = (int)Math.Round(w * x);
                int Y = (int)Math.Round(h * y);
                int W = (int)Math.Round(w * rw);
                int H = (int)Math.Round(h * rh);
                return new Rect(X, Y, W, H);
            }

            // 右下角三档（紧→宽）
            L.Add(R(0.72, 0.86, 0.26, 0.12));
            L.Add(R(0.66, 0.82, 0.32, 0.16));
            L.Add(R(0.60, 0.78, 0.38, 0.20));
            // 下边条 & 右边条
            L.Add(R(0.00, 0.86, 1.00, 0.14));
            L.Add(R(0.80, 0.00, 0.20, 1.00));
            // 全图兜底
            L.Add(new Rect(0, 0, w, h));
            return L;
        }

        private static Rect ClampRect(Rect r, int w, int h)
        {
            int x = Math.Max(0, Math.Min(r.X, w));
            int y = Math.Max(0, Math.Min(r.Y, h));
            int right = Math.Max(0, Math.Min(r.X + r.Width, w));
            int bottom = Math.Max(0, Math.Min(r.Y + r.Height, h));
            int ww = Math.Max(0, right - x);
            int hh = Math.Max(0, bottom - y);
            return new Rect(x, y, ww, hh);
        }

        // --------- OpenCV 预处理（快 & 稳） ----------
        private static Mat PreprocessCv(Mat srcBgr)
        {
            // 灰度
            using var gray = new Mat();
            Cv2.CvtColor(srcBgr, gray, ColorConversionCodes.BGR2GRAY);

            // 轻微降噪（GaussianBlur）
            using var blur = new Mat();
            Cv2.GaussianBlur(gray, blur, new Size(3, 3), 0);

            // 直方图均衡（比 CLAHE 激进程度低）
            using var eq = new Mat();
            Cv2.EqualizeHist(blur, eq);

            // 转回 3 通道，保持 PaddleOCR 输入习惯
            var dstBgr = new Mat();
            Cv2.CvtColor(eq, dstBgr, ColorConversionCodes.GRAY2BGR);

            return dstBgr;
        }


        // --------- 文本清洗 ----------
        private static string Normalize(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return string.Empty;
            s = s.Replace("\r", "").Replace("\n", "").Replace(" ", "").Replace("\t", "");
            var kept = new StringBuilder(s.Length);
            foreach (var ch in s)
            {
                if ((ch >= 0x4e00 && ch <= 0x9fff) || // CJK
                    (ch >= 'A' && ch <= 'Z') ||
                    (ch >= 'a' && ch <= 'z') ||
                    ch == '·' || ch == '・')
                    kept.Append(ch);
            }
            return kept.ToString();
        }

        public void Dispose() => _ocr?.Dispose();
    }
}
