using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using MapWalker.Terrain;

namespace MapWalker.Utilities;

internal static class ImageLoader
{
    public static TileImage? LoadImage(FileInfo file)
    {
        var extension = file.Extension.ToLowerInvariant();
        return extension switch
        {
            ".tga" => LoadTga(file),
            _ => LoadBitmap(file),
        };
    }

    private static TileImage? LoadBitmap(FileInfo file)
    {
        using var bitmap = new Bitmap(file.FullName);
        var rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
        var data = bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        try
        {
            var length = bitmap.Width * bitmap.Height * 4;
            var pixels = new byte[length];
            System.Runtime.InteropServices.Marshal.Copy(data.Scan0, pixels, 0, length);
            for (var i = 0; i < pixels.Length; i += 4)
            {
                (pixels[i + 0], pixels[i + 2]) = (pixels[i + 2], pixels[i + 0]);
            }

            return new TileImage(bitmap.Width, bitmap.Height, pixels);
        }
        finally
        {
            bitmap.UnlockBits(data);
        }
    }

    private static TileImage? LoadTga(FileInfo file)
    {
        using var stream = file.OpenRead();
        using var reader = new BinaryReader(stream);
        var idLength = reader.ReadByte();
        var colorMapType = reader.ReadByte();
        var imageType = reader.ReadByte();
        if (colorMapType != 0 || imageType is not (2 or 3))
        {
            return null;
        }

        reader.BaseStream.Seek(5, SeekOrigin.Current); // color map specification
        reader.BaseStream.Seek(4, SeekOrigin.Current); // x-origin, y-origin
        var width = reader.ReadInt16();
        var height = reader.ReadInt16();
        var pixelDepth = reader.ReadByte();
        var descriptor = reader.ReadByte();
        if (width <= 0 || height <= 0)
        {
            return null;
        }

        if (pixelDepth is not (24 or 32))
        {
            return null;
        }

        if (idLength > 0)
        {
            reader.BaseStream.Seek(idLength, SeekOrigin.Current);
        }

        var bytesPerPixel = pixelDepth / 8;
        var raw = reader.ReadBytes(width * height * bytesPerPixel);
        if (raw.Length != width * height * bytesPerPixel)
        {
            return null;
        }

        var topOrigin = (descriptor & 0x20) != 0;
        var pixels = new byte[width * height * 4];
        for (var y = 0; y < height; y++)
        {
            var srcY = topOrigin ? y : height - 1 - y;
            for (var x = 0; x < width; x++)
            {
                var srcIndex = (srcY * width + x) * bytesPerPixel;
                var dstIndex = (y * width + x) * 4;
                var b = raw[srcIndex + 0];
                var g = raw[srcIndex + 1];
                var r = raw[srcIndex + 2];
                var a = bytesPerPixel == 4 ? raw[srcIndex + 3] : (byte)255;
                pixels[dstIndex + 0] = r;
                pixels[dstIndex + 1] = g;
                pixels[dstIndex + 2] = b;
                pixels[dstIndex + 3] = a;
            }
        }

        return new TileImage(width, height, pixels);
    }
}
