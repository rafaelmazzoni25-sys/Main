using System.Buffers.Binary;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace TerrenoVisualisado.Core;

public sealed class TextureImage
{
    public TextureImage(int width, int height, byte[] pixels)
    {
        Width = width;
        Height = height;
        Pixels = pixels;
    }

    public int Width { get; }
    public int Height { get; }
    public byte[] Pixels { get; }

    public TextureImage Resize(int targetWidth, int targetHeight)
    {
        if (targetWidth <= 0 || targetHeight <= 0)
        {
            throw new ArgumentOutOfRangeException("targetWidth");
        }
        if (targetWidth == Width && targetHeight == Height)
        {
            return this;
        }

        var result = new byte[targetWidth * targetHeight * 4];
        var scaleX = (float)Width / targetWidth;
        var scaleY = (float)Height / targetHeight;
        for (var y = 0; y < targetHeight; y++)
        {
            var srcY = (y + 0.5f) * scaleY - 0.5f;
            for (var x = 0; x < targetWidth; x++)
            {
                var srcX = (x + 0.5f) * scaleX - 0.5f;
                var color = SampleBilinear(srcX, srcY);
                var offset = (y * targetWidth + x) * 4;
                result[offset + 0] = color[0];
                result[offset + 1] = color[1];
                result[offset + 2] = color[2];
                result[offset + 3] = color[3];
            }
        }
        return new TextureImage(targetWidth, targetHeight, result);
    }

    public Bitmap ToBitmap()
    {
        var bitmap = new Bitmap(Width, Height, PixelFormat.Format32bppArgb);
        var data = bitmap.LockBits(new Rectangle(0, 0, Width, Height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
        try
        {
            for (var y = 0; y < Height; y++)
            {
                var srcOffset = y * Width * 4;
                var dstPtr = IntPtr.Add(data.Scan0, y * data.Stride);
                Marshal.Copy(Pixels, srcOffset, dstPtr, Width * 4);
            }
        }
        finally
        {
            bitmap.UnlockBits(data);
        }
        return bitmap;
    }

    public void SavePng(string path)
    {
        using var bitmap = ToBitmap();
        bitmap.Save(path, ImageFormat.Png);
    }

    private byte[] SampleBilinear(float x, float y)
    {
        x = Math.Clamp(x, 0f, Width - 1);
        y = Math.Clamp(y, 0f, Height - 1);
        var x0 = (int)Math.Floor(x);
        var y0 = (int)Math.Floor(y);
        var x1 = Math.Min(x0 + 1, Width - 1);
        var y1 = Math.Min(y0 + 1, Height - 1);
        var tx = x - x0;
        var ty = y - y0;
        Span<float> accum = stackalloc float[4];
        Span<byte> c00 = stackalloc byte[4];
        Span<byte> c10 = stackalloc byte[4];
        Span<byte> c01 = stackalloc byte[4];
        Span<byte> c11 = stackalloc byte[4];
        ReadPixel(x0, y0, c00);
        ReadPixel(x1, y0, c10);
        ReadPixel(x0, y1, c01);
        ReadPixel(x1, y1, c11);
        Span<byte> result = stackalloc byte[4];
        for (var i = 0; i < 4; i++)
        {
            accum[i] = c00[i] * (1 - tx) * (1 - ty)
                + c10[i] * tx * (1 - ty)
                + c01[i] * (1 - tx) * ty
                + c11[i] * tx * ty;
            result[i] = (byte)Math.Clamp((int)Math.Round(accum[i]), 0, 255);
        }
        return result.ToArray();
    }

    private void ReadPixel(int x, int y, Span<byte> destination)
    {
        var offset = (y * Width + x) * 4;
        destination[0] = Pixels[offset + 0];
        destination[1] = Pixels[offset + 1];
        destination[2] = Pixels[offset + 2];
        destination[3] = Pixels[offset + 3];
    }

    public static TextureImage FromRgba(int width, int height, byte[] pixels)
    {
        return new TextureImage(width, height, pixels);
    }
}

internal static class TextureFileLoader
{
    private static readonly string[] ImageExtensions =
    {
        ".ozj",
        ".ozt",
        ".jpg",
        ".jpeg",
        ".png",
        ".tga",
        ".bmp",
    };

    public static bool TryLoad(string path, out TextureImage? image)
    {
        image = null;
        if (!File.Exists(path))
        {
            return false;
        }

        var extension = Path.GetExtension(path);
        if (string.Equals(extension, ".ozj", StringComparison.OrdinalIgnoreCase))
        {
            image = LoadOzj(path);
            return image != null;
        }
        if (string.Equals(extension, ".ozt", StringComparison.OrdinalIgnoreCase))
        {
            image = LoadOzt(path);
            return image != null;
        }
        if (string.Equals(extension, ".tga", StringComparison.OrdinalIgnoreCase))
        {
            image = LoadTga(File.ReadAllBytes(path));
            return image != null;
        }

        using var stream = File.OpenRead(path);
        using var bitmap = (Bitmap)Image.FromStream(stream);
        return TryFromBitmap(bitmap, out image);
    }

    public static IEnumerable<string> EnumerateCandidatePaths(string root, string baseName)
    {
        if (!Directory.Exists(root))
        {
            yield break;
        }

        var normalized = baseName.Replace('\\', '/');
        var variants = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            baseName,
            normalized,
            normalized.ToLowerInvariant(),
            normalized.ToUpperInvariant(),
        };

        foreach (var variant in variants)
        {
            var direct = Path.Combine(root, variant);
            if (File.Exists(direct))
            {
                yield return direct;
            }
            foreach (var ext in ImageExtensions)
            {
                var withExt = Path.Combine(root, variant + ext);
                if (File.Exists(withExt))
                {
                    yield return withExt;
                }
                var alt = Path.Combine(root, variant + "." + ext.TrimStart('.'));
                if (File.Exists(alt))
                {
                    yield return alt;
                }
            }
        }
    }

    private static TextureImage? LoadOzj(string path)
    {
        var data = File.ReadAllBytes(path);
        if (data.Length <= 24)
        {
            return null;
        }
        using var stream = new MemoryStream(data, 24, data.Length - 24, writable: false);
        using var bitmap = (Bitmap)Image.FromStream(stream);
        return TryFromBitmap(bitmap, out var image) ? image : null;
    }

    private static TextureImage? LoadOzt(string path)
    {
        var data = File.ReadAllBytes(path);
        if (data.Length <= 4)
        {
            return null;
        }
        var tgaData = data.AsSpan(4).ToArray();
        return LoadTga(tgaData);
    }

    private static TextureImage? LoadTga(byte[] data)
    {
        if (data.Length < 18)
        {
            return null;
        }
        var idLength = data[0];
        var colorMapType = data[1];
        var imageType = data[2];
        if (colorMapType != 0 || (imageType != 2 && imageType != 3 && imageType != 10))
        {
            return null;
        }
        var width = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(12));
        var height = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(14));
        var bpp = data[16];
        var descriptor = data[17];
        var originTop = (descriptor & 0x20) != 0;
        var originLeft = (descriptor & 0x10) == 0;
        var offset = 18 + idLength;
        if (width <= 0 || height <= 0)
        {
            return null;
        }
        if (imageType == 10)
        {
            // RLE compressed: expand
            var pixels = new byte[width * height * 4];
            var index = 0;
            while (index < pixels.Length && offset < data.Length)
            {
                var packet = data[offset++];
                var count = (packet & 0x7F) + 1;
                if ((packet & 0x80) != 0)
                {
                    if (!ReadBgr(data, ref offset, bpp, out var color))
                    {
                        break;
                    }
                    for (var i = 0; i < count && index < pixels.Length; i++)
                    {
                        WritePixel(pixels, ref index, color);
                    }
                }
                else
                {
                    for (var i = 0; i < count && index < pixels.Length; i++)
                    {
                        if (!ReadBgr(data, ref offset, bpp, out var color))
                        {
                            index = pixels.Length;
                            break;
                        }
                        WritePixel(pixels, ref index, color);
                    }
                }
            }
            ReorderTga(pixels, width, height, originLeft, originTop);
            return new TextureImage(width, height, pixels);
        }
        else
        {
            var expected = width * height * (bpp / 8);
            if (offset + expected > data.Length)
            {
                return null;
            }
            var pixels = new byte[width * height * 4];
            var index = 0;
            for (var i = 0; i < width * height; i++)
            {
                if (!ReadBgr(data, ref offset, bpp, out var color))
                {
                    return null;
                }
                WritePixel(pixels, ref index, color);
            }
            ReorderTga(pixels, width, height, originLeft, originTop);
            return new TextureImage(width, height, pixels);
        }
    }

    private static void ReorderTga(byte[] pixels, int width, int height, bool originLeft, bool originTop)
    {
        if (!originLeft)
        {
            FlipHorizontal(pixels, width, height);
        }
        if (!originTop)
        {
            FlipVertical(pixels, width, height);
        }
    }

    private static void FlipHorizontal(byte[] pixels, int width, int height)
    {
        var stride = width * 4;
        for (var y = 0; y < height; y++)
        {
            var row = y * stride;
            for (var x = 0; x < width / 2; x++)
            {
                var left = row + x * 4;
                var right = row + (width - 1 - x) * 4;
                for (var i = 0; i < 4; i++)
                {
                    (pixels[left + i], pixels[right + i]) = (pixels[right + i], pixels[left + i]);
                }
            }
        }
    }

    private static void FlipVertical(byte[] pixels, int width, int height)
    {
        var stride = width * 4;
        for (var y = 0; y < height / 2; y++)
        {
            var top = y * stride;
            var bottom = (height - 1 - y) * stride;
            for (var x = 0; x < stride; x++)
            {
                (pixels[top + x], pixels[bottom + x]) = (pixels[bottom + x], pixels[top + x]);
            }
        }
    }

    private static bool ReadBgr(byte[] data, ref int offset, int bpp, out (byte B, byte G, byte R, byte A) color)
    {
        color = default;
        var bytesPerPixel = bpp / 8;
        if (bytesPerPixel < 3 || offset + bytesPerPixel > data.Length)
        {
            return false;
        }
        var b = data[offset + 0];
        var g = data[offset + 1];
        var r = data[offset + 2];
        byte a = 255;
        if (bytesPerPixel >= 4)
        {
            a = data[offset + 3];
        }
        offset += bytesPerPixel;
        color = (b, g, r, a);
        return true;
    }

    private static void WritePixel(byte[] pixels, ref int index, (byte B, byte G, byte R, byte A) color)
    {
        if (index + 4 > pixels.Length)
        {
            return;
        }
        pixels[index++] = color.R;
        pixels[index++] = color.G;
        pixels[index++] = color.B;
        pixels[index++] = color.A;
    }

    private static bool TryFromBitmap(Bitmap bitmap, out TextureImage? image)
    {
        image = null;
        var rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
        var data = bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        try
        {
            var buffer = new byte[bitmap.Width * bitmap.Height * 4];
            for (var y = 0; y < bitmap.Height; y++)
            {
                var srcPtr = IntPtr.Add(data.Scan0, y * data.Stride);
                Marshal.Copy(srcPtr, buffer, y * bitmap.Width * 4, bitmap.Width * 4);
            }
            image = new TextureImage(bitmap.Width, bitmap.Height, buffer);
            return true;
        }
        finally
        {
            bitmap.UnlockBits(data);
        }
    }
}
