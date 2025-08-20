using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace ImageRenamer.Services;

public sealed class NameMatcher
{
    private readonly List<string> _fullNames;
    private readonly SimilarMap _map;
    private readonly bool _singleFallback;

    public NameMatcher(List<string> names, SimilarMap map, bool singleFallback)
    {
        _fullNames = names;
        _map = map;
        _singleFallback = singleFallback;
    }

    public static async System.Threading.Tasks.Task<List<string>> LoadNamesAsync(string path)
    {
        var list = new List<string>();
        using var sr = new StreamReader(path, Encoding.UTF8);
        while (!sr.EndOfStream)
        {
            var line = (await sr.ReadLineAsync())?.Trim();
            if (!string.IsNullOrEmpty(line) && line!.Length is >= 2 and <= 4)
                list.Add(line!);
        }
        return list;
    }

    public static string ToSafeFilename(string name)
    {
        var invalid = System.IO.Path.GetInvalidFileNameChars();
        var sb = new StringBuilder();
        foreach (var ch in name)
            sb.Append(invalid.Contains(ch) ? '#' : ch);
        var s = sb.ToString();
        if (s.Length > 220) s = s[..220];
        return s;
    }

    public string? FindBest(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return null;

        string onlyCn = ExtractCn(raw);
        string replaced = _map.Apply(onlyCn);

        // 1) 直接包含匹配
        foreach (var name in _fullNames)
        {
            if (replaced.Contains(name) || onlyCn.Contains(name))
                return name;
        }

        // 2) 最长公共子串（阈值 >= 2）
        string best = "";
        foreach (var name in _fullNames)
        {
            var lcs = LongestCommonSubstring(replaced, name);
            if (lcs.Length > best.Length) best = name;
        }
        if (best.Length > 0 && LongestCommonSubstring(replaced, best).Length >= 2)
            return best;

        // 3) 单字兜底：统计共享汉字出现次数（同你现在的思路:contentReference[oaicite:4]{index=4}）
        if (_singleFallback)
        {
            var scores = new Dictionary<string, int>();
            foreach (var ch in replaced)
            {
                foreach (var n in _fullNames)
                    if (n.Contains(ch))
                        scores[n] = scores.TryGetValue(n, out var v) ? v + 1 : 1;
            }
            if (scores.Count > 0)
                return scores.OrderByDescending(kv => kv.Value)
                             .ThenBy(kv => _fullNames.IndexOf(kv.Key))
                             .First().Key;
        }
        return null;
    }

    private static string ExtractCn(string s)
    {
        var sb = new StringBuilder();
        foreach (var ch in s)
            if (ch >= 0x4e00 && ch <= 0x9fff) sb.Append(ch);
        return sb.ToString();
    }

    private static string LongestCommonSubstring(string a, string b)
    {
        if (a.Length == 0 || b.Length == 0) return "";
        int[,] dp = new int[a.Length + 1, b.Length + 1];
        int maxLen = 0, end = 0;
        for (int i = 1; i <= a.Length; i++)
        {
            for (int j = 1; j <= b.Length; j++)
            {
                if (a[i - 1] == b[j - 1])
                {
                    dp[i, j] = dp[i - 1, j - 1] + 1;
                    if (dp[i, j] > maxLen)
                    {
                        maxLen = dp[i, j];
                        end = i;
                    }
                }
            }
        }
        return maxLen > 0 ? a.Substring(end - maxLen, maxLen) : "";
    }
}
