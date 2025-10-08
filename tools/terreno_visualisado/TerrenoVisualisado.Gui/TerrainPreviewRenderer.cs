using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using TerrenoVisualisado.Core;

namespace TerrenoVisualisado.Gui;

internal enum PreviewMode
{
    Height,
    Layer1,
    Layer2,
    Alpha,
    Attributes,
}

internal static class TerrainPreviewRenderer
{
    public static Bitmap Render(WorldData world, PreviewMode mode, bool overlayObjects)
    {
        var size = WorldLoader.TerrainSize;
        if (mode == PreviewMode.Layer1 && world.Visual?.CompositeTexture is { } composite)
        {
            var bitmap = composite.ToBitmap();
            if (overlayObjects)
            {
                OverlayObjects(world, bitmap);
            }
            return bitmap;
        }
        var bitmap = new Bitmap(size, size, PixelFormat.Format32bppArgb);
        var rect = new Rectangle(0, 0, size, size);
        var data = bitmap.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
        try
        {
            var stride = data.Stride;
            var buffer = new byte[stride * size];
            switch (mode)
            {
                case PreviewMode.Height:
                    RenderHeight(world.Terrain.Height, buffer, stride, size);
                    break;
                case PreviewMode.Layer1:
                    RenderTileLayer(world.Terrain.Layer1, buffer, stride, size);
                    break;
                case PreviewMode.Layer2:
                    RenderTileLayer(world.Terrain.Layer2, buffer, stride, size);
                    break;
                case PreviewMode.Alpha:
                    RenderAlpha(world.Terrain.Alpha, buffer, stride, size);
                    break;
                case PreviewMode.Attributes:
                    RenderAttributes(world.Terrain.Attributes, buffer, stride, size);
                    break;
            }

            Marshal.Copy(buffer, 0, data.Scan0, buffer.Length);
        }
        finally
        {
            bitmap.UnlockBits(data);
        }

        if (overlayObjects)
        {
            OverlayObjects(world, bitmap);
        }

        return bitmap;
    }

    private static void OverlayObjects(WorldData world, Bitmap bitmap)
    {
        var size = WorldLoader.TerrainSize;
        using var g = Graphics.FromImage(bitmap);
        using var brush = new SolidBrush(Color.FromArgb(192, Color.Red));
        foreach (var obj in world.Objects)
        {
            var px = obj.RawPosition.X / 100f;
            var py = obj.RawPosition.Y / 100f;
            var x = Math.Clamp((int)MathF.Round(px), 0, size - 1);
            var y = Math.Clamp((int)MathF.Round(py), 0, size - 1);
            var screenY = size - 1 - y;
            g.FillEllipse(brush, x - 2, screenY - 2, 4, 4);
        }
    }

    private static void RenderHeight(float[] height, byte[] buffer, int stride, int size)
    {
        var min = float.MaxValue;
        var max = float.MinValue;
        foreach (var value in height)
        {
            if (!float.IsFinite(value))
            {
                continue;
            }
            if (value < min) min = value;
            if (value > max) max = value;
        }

        if (min > max)
        {
            min = 0;
            max = 1;
        }

        var range = Math.Max(max - min, 0.01f);
        for (var y = 0; y < size; y++)
        {
            var row = buffer.AsSpan(y * stride, stride);
            for (var x = 0; x < size; x++)
            {
                var index = y * size + x;
                var value = height[index];
                if (!float.IsFinite(value))
                {
                    value = min;
                }
                var t = (value - min) / range;
                var color = SampleHeightGradient(t);
                var offset = x * 4;
                row[offset + 0] = color.B;
                row[offset + 1] = color.G;
                row[offset + 2] = color.R;
                row[offset + 3] = 255;
            }
        }
    }

    private static void RenderTileLayer(byte[] data, byte[] buffer, int stride, int size)
    {
        for (var y = 0; y < size; y++)
        {
            var row = buffer.AsSpan(y * stride, stride);
            for (var x = 0; x < size; x++)
            {
                var value = data[y * size + x];
                var color = ColorFromTile(value);
                var offset = x * 4;
                row[offset + 0] = color.B;
                row[offset + 1] = color.G;
                row[offset + 2] = color.R;
                row[offset + 3] = 255;
            }
        }
    }

    private static void RenderAlpha(float[] alpha, byte[] buffer, int stride, int size)
    {
        for (var y = 0; y < size; y++)
        {
            var row = buffer.AsSpan(y * stride, stride);
            for (var x = 0; x < size; x++)
            {
                var value = alpha[y * size + x];
                var clamped = (byte)Math.Clamp((int)MathF.Round(value * 255f), 0, 255);
                var offset = x * 4;
                row[offset + 0] = clamped;
                row[offset + 1] = clamped;
                row[offset + 2] = clamped;
                row[offset + 3] = 255;
            }
        }
    }

    private static void RenderAttributes(ushort[] attributes, byte[] buffer, int stride, int size)
    {
        for (var y = 0; y < size; y++)
        {
            var row = buffer.AsSpan(y * stride, stride);
            for (var x = 0; x < size; x++)
            {
                var value = attributes[y * size + x];
                var color = ColorFromAttribute(value);
                var offset = x * 4;
                row[offset + 0] = color.B;
                row[offset + 1] = color.G;
                row[offset + 2] = color.R;
                row[offset + 3] = 255;
            }
        }
    }

    private static Color SampleHeightGradient(float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        if (t < 0.33f)
        {
            return Lerp(Color.DarkBlue, Color.CornflowerBlue, t / 0.33f);
        }
        if (t < 0.66f)
        {
            return Lerp(Color.ForestGreen, Color.YellowGreen, (t - 0.33f) / 0.33f);
        }
        return Lerp(Color.SaddleBrown, Color.WhiteSmoke, (t - 0.66f) / 0.34f);
    }

    private static Color ColorFromTile(byte value)
    {
        var hue = (value / 255f) * 360f;
        return ColorFromHsv(hue, 0.6f, 0.9f);
    }

    private static Color ColorFromAttribute(ushort value)
    {
        var hue = ((value * 37) % 360);
        var saturation = 0.5f + ((value % 5) * 0.1f);
        return ColorFromHsv(hue, Math.Clamp(saturation, 0f, 1f), 0.85f);
    }

    private static Color Lerp(Color a, Color b, float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        var r = (byte)(a.R + (b.R - a.R) * t);
        var g = (byte)(a.G + (b.G - a.G) * t);
        var bCh = (byte)(a.B + (b.B - a.B) * t);
        return Color.FromArgb(r, g, bCh);
    }

    private static Color ColorFromHsv(double hue, double saturation, double value)
    {
        saturation = Math.Clamp(saturation, 0.0, 1.0);
        value = Math.Clamp(value, 0.0, 1.0);
        var c = value * saturation;
        var x = c * (1 - Math.Abs((hue / 60.0 % 2) - 1));
        var m = value - c;

        double r = 0, g = 0, b = 0;
        if (hue < 60)
        {
            r = c; g = x; b = 0;
        }
        else if (hue < 120)
        {
            r = x; g = c; b = 0;
        }
        else if (hue < 180)
        {
            r = 0; g = c; b = x;
        }
        else if (hue < 240)
        {
            r = 0; g = x; b = c;
        }
        else if (hue < 300)
        {
            r = x; g = 0; b = c;
        }
        else
        {
            r = c; g = 0; b = x;
        }

        var red = (byte)Math.Clamp((r + m) * 255.0, 0, 255);
        var green = (byte)Math.Clamp((g + m) * 255.0, 0, 255);
        var blue = (byte)Math.Clamp((b + m) * 255.0, 0, 255);
        return Color.FromArgb(red, green, blue);
    }
}
