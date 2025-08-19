using System.Collections.Generic;
using System.IO;
using System.Text;

namespace ImageRenamer.Services;

public sealed class SimilarMap
{
    private readonly Dictionary<string, string> _map = new();
    public int Count => _map.Count;

    public void TryLoadFromFile(string path)
    {
        if (!File.Exists(path)) return;
        foreach (var line in File.ReadAllLines(path, Encoding.UTF8))
        {
            var s = line.Trim();
            if (string.IsNullOrEmpty(s)) continue;
            var idx = s.IndexOf(':');
            if (idx > 0)
            {
                var key = s[..idx].Trim();
                var val = s[(idx + 1)..].Trim();
                if (!string.IsNullOrEmpty(key) && !string.IsNullOrEmpty(val))
                    _map[key] = val;
            }
        }
    }

    public string Apply(string text)
    {
        if (_map.Count == 0) return text;
        var sb = new StringBuilder();
        int i = 0;
        while (i < text.Length)
        {
            bool matched = false;
            if (i + 1 < text.Length)
            {
                var pair = text.Substring(i, 2);
                if (_map.TryGetValue(pair, out var rep2))
                {
                    sb.Append(rep2);
                    i += 2;
                    matched = true;
                }
            }
            if (!matched)
            {
                var ch = text[i].ToString();
                if (_map.TryGetValue(ch, out var rep1))
                    sb.Append(rep1);
                else
                    sb.Append(ch);
                i++;
            }
        }
        return sb.ToString();
    }
}
