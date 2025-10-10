using System;
using System.Collections.Generic;
using OpenTK.Mathematics;

namespace MapWalker.Terrain;

internal sealed class TileAtlas
{
    public TileAtlas(IReadOnlyList<TileImage> images)
    {
        if (images.Count == 0)
        {
            throw new ArgumentException("É necessário ao menos uma textura de tile.", nameof(images));
        }

        TileSize = ComputeTileSize(images);
        Columns = (int)Math.Ceiling(Math.Sqrt(images.Count));
        Rows = (int)Math.Ceiling(images.Count / (float)Columns);
        Width = Columns * TileSize;
        Height = Rows * TileSize;
        Pixels = new byte[Width * Height * 4];
        for (var i = 0; i < images.Count; i++)
        {
            var col = i % Columns;
            var row = i / Columns;
            Blit(images[i], col * TileSize, row * TileSize);
        }
    }

    public int TileSize { get; }
    public int Columns { get; }
    public int Rows { get; }
    public int Width { get; }
    public int Height { get; }
    public byte[] Pixels { get; }

    public Vector2 AtlasStep => new(1f / Columns, 1f / Rows);

    private static int ComputeTileSize(IReadOnlyList<TileImage> images)
    {
        var size = 1;
        foreach (var image in images)
        {
            size = Math.Max(size, Math.Max(image.Width, image.Height));
        }

        // ensure power-of-two size for better sampling
        var result = 1;
        while (result < size)
        {
            result <<= 1;
        }

        return Math.Max(result, 4);
    }

    private void Blit(TileImage image, int offsetX, int offsetY)
    {
        for (var y = 0; y < TileSize; y++)
        {
            var srcY = TileSize > 1 ? (int)Math.Round(y / (float)(TileSize - 1) * (image.Height - 1)) : 0;
            for (var x = 0; x < TileSize; x++)
            {
                var srcX = TileSize > 1 ? (int)Math.Round(x / (float)(TileSize - 1) * (image.Width - 1)) : 0;
                var srcIndex = (srcY * image.Width + srcX) * 4;
                var dstIndex = ((offsetY + y) * Width + (offsetX + x)) * 4;
                Pixels[dstIndex + 0] = image.Pixels[srcIndex + 0];
                Pixels[dstIndex + 1] = image.Pixels[srcIndex + 1];
                Pixels[dstIndex + 2] = image.Pixels[srcIndex + 2];
                Pixels[dstIndex + 3] = image.Pixels[srcIndex + 3];
            }
        }
    }
}
