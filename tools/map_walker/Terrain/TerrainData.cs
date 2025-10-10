using System;
using System.Collections.Generic;
using OpenTK.Mathematics;

namespace MapWalker.Terrain;

internal sealed class TerrainData
{
    public const int TerrainSize = 256;
    public const float TerrainScale = 100f;
    public const float MinExtendedHeight = -500f;

    private readonly float[,] _heights;
    private readonly byte[,]? _attributes;
    private readonly byte[,]? _tileLayer1;
    private readonly byte[,]? _tileLayer2;
    private readonly float[,]? _tileAlpha;
    private readonly int _size;

    public TerrainData(
        MapDescriptor descriptor,
        float[,] heights,
        byte[,]? attributes,
        IReadOnlyList<ObjectInstance> objects,
        byte[,]? tileLayer1,
        byte[,]? tileLayer2,
        float[,]? tileAlpha,
        IReadOnlyList<TileImage> tileImages)
    {
        Descriptor = descriptor;
        _heights = heights;
        _attributes = attributes;
        Objects = objects;
        _tileLayer1 = tileLayer1;
        _tileLayer2 = tileLayer2;
        _tileAlpha = tileAlpha;
        TileImages = tileImages;
        _size = heights.GetLength(0);
    }

    public MapDescriptor Descriptor { get; }

    public IReadOnlyList<ObjectInstance> Objects { get; }

    public IReadOnlyList<TileImage> TileImages { get; }

    public float[,] Heights => _heights;

    public int Size => _size;

    public float SampleHeight(float x, float z)
    {
        var scale = TerrainScale;
        var xf = Math.Clamp(x / scale, 0f, _size - 1.0001f);
        var zf = Math.Clamp(z / scale, 0f, _size - 1.0001f);
        var x0 = (int)MathF.Floor(xf);
        var x1 = Math.Min(x0 + 1, _size - 1);
        var z0 = (int)MathF.Floor(zf);
        var z1 = Math.Min(z0 + 1, _size - 1);
        var xd = xf - x0;
        var zd = zf - z0;
        var h00 = _heights[z0, x0];
        var h01 = _heights[z1, x0];
        var h10 = _heights[z0, x1];
        var h11 = _heights[z1, x1];
        var h0 = MathF.Lerp(h00, h10, xd);
        var h1 = MathF.Lerp(h01, h11, xd);
        return MathF.Lerp(h0, h1, zd);
    }

    public bool IsWalkable(float x, float z)
    {
        if (_attributes is null)
        {
            return true;
        }

        var scale = TerrainScale;
        var xi = (int)(x / scale);
        var zi = (int)(z / scale);
        if (xi < 0 || zi < 0 || xi >= _size || zi >= _size)
        {
            return true;
        }

        var attr = _attributes[zi, xi];
        const byte TwNoMove = 0x04;
        const byte TwNoGround = 0x08;
        const byte TwHeight = 0x40;
        if ((attr & (TwNoMove | TwNoGround)) != 0)
        {
            return false;
        }

        if ((attr & TwHeight) != 0)
        {
            return false;
        }

        return true;
    }

    public float WorldExtent => (_size - 1) * TerrainScale;

    public bool HasTileMapping => _tileLayer1 is not null
        && _tileLayer2 is not null
        && _tileAlpha is not null
        && TileImages.Count > 0;

    public (int Layer1, int Layer2, float Alpha)? TileIndicesAt(float x, float z)
    {
        if (!HasTileMapping)
        {
            return null;
        }

        var xi = (int)(x / TerrainScale);
        var zi = (int)(z / TerrainScale);
        if (xi < 0 || zi < 0 || xi >= _size || zi >= _size)
        {
            return null;
        }

        return (
            _tileLayer1![zi, xi],
            _tileLayer2![zi, xi],
            _tileAlpha![zi, xi]
        );
    }

    public byte[,]? TileLayer1 => _tileLayer1;
    public byte[,]? TileLayer2 => _tileLayer2;
    public float[,]? TileAlpha => _tileAlpha;
}
