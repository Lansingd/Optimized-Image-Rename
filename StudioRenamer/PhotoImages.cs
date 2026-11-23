using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace StudioRenamer;

internal static class PhotoImages
{
    private static readonly Rect WatermarkBounds = new(0.72, 0.925, 0.28, 0.075);
    private static readonly Rect CertificateBounds = new(0.44, 0.47, 0.47, 0.37);

    public static BitmapSource Read(string path, int longestEdge, bool watermark = false) =>
        ReadRegion(path, longestEdge, watermark ? WatermarkBounds : new Rect(0, 0, 1, 1), watermark ? 768 : 1);

    public static BitmapSource ReadCertificateRegion(string path) => ReadRegion(path, 1800, CertificateBounds, 1056);

    private static BitmapSource ReadRegion(string path, int longestEdge, Rect relative, int minimumEdge)
    {
        using var stream = File.OpenRead(path);
        var frame = BitmapDecoder.Create(stream, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnDemand).Frames[0];
        int orientation = 1;
        try
        {
            if (frame.Metadata is BitmapMetadata metadata)
            {
                object? value = metadata.GetQuery("/app1/ifd/{ushort=274}") ??
                    metadata.GetQuery("/ifd/{ushort=274}");
                if (value is ushort exifOrientation) orientation = exifOrientation;
            }
        }
        catch (NotSupportedException) { }

        int w = frame.PixelWidth, h = frame.PixelHeight;
        bool swap = orientation is >= 5 and <= 8;
        int orientedWidth = swap ? h : w;
        int orientedHeight = swap ? w : h;
        var crop = new Rect(orientedWidth * relative.X, orientedHeight * relative.Y,
            orientedWidth * relative.Width, orientedHeight * relative.Height);
        double edge = Math.Max(crop.Width, crop.Height);
        double scale = Math.Clamp(edge, Math.Min(minimumEdge, longestEdge), longestEdge) / edge;
        int width = Math.Max(1, (int)Math.Round(crop.Width * scale));
        int height = Math.Max(1, (int)Math.Round(crop.Height * scale));
        Matrix transform = orientation switch
        {
            2 => new Matrix(-1, 0, 0, 1, w, 0),
            3 => new Matrix(-1, 0, 0, -1, w, h),
            4 => new Matrix(1, 0, 0, -1, 0, h),
            5 => new Matrix(0, 1, 1, 0, 0, 0),
            6 => new Matrix(0, 1, -1, 0, h, 0),
            7 => new Matrix(0, -1, -1, 0, h, w),
            8 => new Matrix(0, -1, 1, 0, 0, w),
            _ => Matrix.Identity
        };
        transform.Translate(-crop.X, -crop.Y);
        transform.Scale(width / crop.Width, height / crop.Height);
        return Render(frame, new Rect(0, 0, w, h), width, height, transform);
    }

    // Use the checkpoint's 16px patches, 2x spatial merge and pixel limits.
    public static byte[] Prepare(BitmapSource source, bool lowerRight)
    {
        if (lowerRight) source = LowerRight(source);
        const int grid = 32, minimumPixels = 65_536, maximumPixels = 16_777_216;
        int w = source.PixelWidth, h = source.PixelHeight;
        if (Math.Max(w, h) / (double)Math.Min(w, h) > 200)
            throw new InvalidDataException("图片过于狭长，请裁成较短的图片再试。");

        int width = Math.Max(grid, (int)Math.Round(w / (double)grid) * grid);
        int height = Math.Max(grid, (int)Math.Round(h / (double)grid) * grid);
        if (width * height > maximumPixels)
        {
            double scale = Math.Sqrt(maximumPixels / ((double)w * h));
            width = Math.Max(grid, (int)Math.Floor(w * scale / grid) * grid);
            height = Math.Max(grid, (int)Math.Floor(h * scale / grid) * grid);
        }
        else if (width * height < minimumPixels)
        {
            double scale = Math.Sqrt(minimumPixels / ((double)w * h));
            width = (int)Math.Ceiling(w * scale / grid) * grid;
            height = (int)Math.Ceiling(h * scale / grid) * grid;
        }

        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(Render(source, new Rect(0, 0, width, height), width, height, Matrix.Identity)));
        using var stream = new MemoryStream();
        encoder.Save(stream);
        return stream.ToArray();
    }

    public static BitmapSource LowerRight(BitmapSource source)
    {
        int left = source.PixelWidth * 45 / 100;
        int top = source.PixelHeight * 60 / 100;
        return new CroppedBitmap(source, new Int32Rect(left, top, source.PixelWidth - left, source.PixelHeight - top));
    }

    private static BitmapSource Render(BitmapSource source, Rect imageBounds, int width, int height, Matrix transform)
    {
        var visual = new DrawingVisual();
        RenderOptions.SetBitmapScalingMode(visual, BitmapScalingMode.HighQuality);
        using (DrawingContext drawing = visual.RenderOpen())
        {
            drawing.DrawRectangle(Brushes.White, null, new Rect(0, 0, width, height));
            drawing.PushTransform(new MatrixTransform(transform));
            drawing.DrawImage(source, imageBounds);
            drawing.Pop();
        }

        var bitmap = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Pbgra32);
        bitmap.Render(visual);
        bitmap.Freeze();
        return bitmap;
    }
}
