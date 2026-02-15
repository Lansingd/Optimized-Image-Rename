using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace ImageRenamer.Utils;

public static class Downloader
{
    // 高精度模型（比 fast 更准）：~26MB
    private const string ChiSimBest = "https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata";

    public static async Task<string> EnsureChiSimAsync(Action<string> log)
    {
        string baseDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ImageRenamer", "tessdata"
        );
        Directory.CreateDirectory(baseDir);

        string modelPath = Path.Combine(baseDir, "chi_sim.traineddata");
        if (!File.Exists(modelPath))
        {
            log("首次运行：正在下载 chi_sim（tessdata_best，高精度）模型…");
            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
            var bytes = await http.GetByteArrayAsync(ChiSimBest);
            await File.WriteAllBytesAsync(modelPath, bytes);
            log("chi_sim（best）模型下载完成。");
        }
        return baseDir;
    }
}
