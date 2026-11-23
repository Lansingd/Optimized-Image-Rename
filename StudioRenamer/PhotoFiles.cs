using System.ComponentModel;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace StudioRenamer;

internal sealed class Photo(string path, string relativePath) : INotifyPropertyChanged
{
    public string Source { get; } = path;
    public string RelativePath { get; } = relativePath;
    public string? Output { get; private set; }
    public string Reading { get; private set; } = "";
    public string Status { get; private set; } = "待识别";
    public string Detail { get; private set; } = "可先识别，也可以直接输入姓名。";
    public bool NeedsScan { get; private set; } = true;
    private string name = "";
    private bool included;

    public string Name
    {
        get => name;
        set
        {
            if (name == value || Output != null) return;
            name = value;
            included = PhotoFiles.IsName(name.Trim());
            NeedsScan = false;
            Status = included ? "已编辑" : "待确认";
            Detail = included ? "已使用你填写的姓名。" : "请输入有效的姓名。";
            Changed();
        }
    }

    public bool Included
    {
        get => included;
        set
        {
            if (Output != null || included == value) return;
            included = value;
            Changed();
        }
    }

    public void Scanned(string reading, string suggestion, bool selected, string status, string detail, bool retry = false)
    {
        Reading = reading;
        name = suggestion;
        included = selected;
        NeedsScan = retry;
        Status = status;
        Detail = detail;
        Changed();
    }

    public void Saved(string output, string status)
    {
        Output = output;
        Status = status;
        Detail = output;
        included = false;
        Changed();
    }

    public void SaveFailed(string message)
    {
        Status = "写入失败";
        Detail = message;
        Changed();
    }

    public event PropertyChangedEventHandler? PropertyChanged;
    private void Changed() => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(null));
}

internal static class PhotoFiles
{
    public const string ArchiveDirectoryName = "已整理";
    private static readonly HashSet<string> Extensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".tif", ".tiff", ".webp"
    };
    private static readonly Regex NamePattern = new(
        @"^[\p{IsCJKUnifiedIdeographs}\p{IsCJKUnifiedIdeographsExtensionA}\p{IsCJKCompatibilityIdeographs}A-Za-z\p{IsLatin-1Supplement}\p{IsLatinExtended-A}·・.'’ -]{1,48}$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
    {
        "CON", "PRN", "AUX", "NUL", "COM", "LPT"
    };

    public static bool IsName(string value) =>
        NamePattern.IsMatch(value) && value.Any(char.IsLetter) &&
        char.IsLetter(value[0]) && char.IsLetter(value[^1]) &&
        !Reserved.Contains(value.Split('.')[0].TrimEnd());

    public static Photo[] Scan(string folder)
    {
        string root = Path.GetFullPath(folder);
        var directories = new Stack<DirectoryInfo>();
        var photos = new List<Photo>();
        directories.Push(new DirectoryInfo(root));

        while (directories.TryPop(out DirectoryInfo? directory))
        {
            foreach (FileInfo file in directory.EnumerateFiles())
            {
                if (Extensions.Contains(file.Extension))
                    photos.Add(new Photo(file.FullName, Path.GetRelativePath(root, file.FullName)));
            }

            foreach (DirectoryInfo child in directory.EnumerateDirectories())
            {
                if (child.Name != ArchiveDirectoryName && !child.Attributes.HasFlag(FileAttributes.ReparsePoint))
                    directories.Push(child);
            }
        }

        return photos.OrderBy(photo => photo.RelativePath, StringComparer.CurrentCultureIgnoreCase).ToArray();
    }

    public static HashSet<string> ReadRoster(string path)
    {
        var bytes = File.ReadAllBytes(path);
        string text;
        if (bytes.AsSpan().StartsWith(new byte[] { 0xff, 0xfe }))
            text = Encoding.Unicode.GetString(bytes);
        else if (bytes.AsSpan().StartsWith(new byte[] { 0xfe, 0xff }))
            text = Encoding.BigEndianUnicode.GetString(bytes);
        else
        {
            try { text = new UTF8Encoding(false, true).GetString(bytes); }
            catch (DecoderFallbackException)
            {
                Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
                text = Encoding.GetEncoding("GB18030").GetString(bytes);
            }
        }
        return text.TrimStart('\ufeff').Split(['\r', '\n', ',', '，', '、', '\t', ';', '；'],
                StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(name => name is not ("姓名" or "学生姓名" or "名字") && IsName(name))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
    }

    public static string[] RosterSuggestions(string[] readings, HashSet<string> roster)
    {
        if (readings.Length == 0 || roster.Count == 0 ||
            (readings.Length == 1 && roster.Contains(readings[0])))
            return [];

        var exact = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var similar = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string reading in readings)
        {
            if (roster.TryGetValue(reading, out string? rosterName))
                exact.Add(rosterName);
            else
            {
                foreach (string candidate in roster)
                {
                    if (OneEditApart(reading, candidate)) similar.Add(candidate);
                }
            }
        }

        similar.ExceptWith(exact);
        return exact.Order(StringComparer.CurrentCulture).Concat(similar.Order(StringComparer.CurrentCulture)).ToArray();
    }

    private static bool OneEditApart(string first, string second)
    {
        string shorter = first.Length <= second.Length ? first : second;
        string longer = first.Length <= second.Length ? second : first;
        if (shorter.Length < 2 || longer.Length - shorter.Length > 1) return false;

        int difference = 0;
        while (difference < shorter.Length && char.ToUpperInvariant(shorter[difference]) == char.ToUpperInvariant(longer[difference]))
            difference++;

        // A different character, or a character added to or missing from the OCR result.
        int shorterTail = shorter.Length == longer.Length ? difference + 1 : difference;
        return difference < longer.Length &&
            shorter.AsSpan(shorterTail).Equals(longer.AsSpan(difference + 1), StringComparison.OrdinalIgnoreCase);
    }

    public static string Write(string source, string name, string? archiveRoot)
    {
        if (!IsName(name)) throw new ArgumentException("请输入有效的姓名。");
        string folder = archiveRoot == null ? Path.GetDirectoryName(source)! : Path.Combine(archiveRoot, name);
        string extension = Path.GetExtension(source);
        string firstPath = Path.Combine(folder, name + extension);
        if (archiveRoot == null && StringComparer.OrdinalIgnoreCase.Equals(firstPath, source))
        {
            if (!File.Exists(source)) throw new FileNotFoundException("找不到原图。", source);
            if (!StringComparer.Ordinal.Equals(firstPath, source)) File.Move(source, firstPath);
            return firstPath;
        }

        string? temporary = null;
        try
        {
            if (archiveRoot != null)
            {
                Directory.CreateDirectory(folder);
                temporary = Path.Combine(folder, $".{Guid.NewGuid():N}.tmp");
                File.Copy(source, temporary, false);
            }

            for (int suffix = 1; suffix <= 100_000; suffix++)
            {
                string target = suffix == 1 ? firstPath : Path.Combine(folder, $"{name}_{suffix}{extension}");
                try
                {
                    File.Move(temporary ?? source, target, false);
                    temporary = null;
                    return target;
                }
                catch (IOException) when (File.Exists(target) || Directory.Exists(target))
                {
                    // Another photo may already be filed under this name.
                }
            }

            throw new IOException("该姓名已有过多重名文件。");
        }
        finally
        {
            if (temporary != null) File.Delete(temporary);
        }
    }
}
