using System;
using System.Numerics;

namespace TerrenoVisualisado.Core;

public sealed class TerrainMesh
{
    public TerrainMesh(float[] vertices, uint[] indices, Vector3 boundsMin, Vector3 boundsMax, int vertexStride)
    {
        Vertices = vertices;
        Indices = indices;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        VertexStride = vertexStride;
    }

    public float[] Vertices { get; }
    public uint[] Indices { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
    public int VertexStride { get; }
}

public static class TerrainMeshBuilder
{
    private const int BaseVertexStride = 8;

    public static TerrainMesh Build(TerrainData terrain, uint[]? materialFlagsPerTile = null)
    {
        var size = WorldLoader.TerrainSize;
        if (terrain.Height.Length != size * size)
        {
            throw new ArgumentException("Height map inválido", nameof(terrain));
        }

        var vertexCount = size * size;
        var includeMaterialFlags = materialFlagsPerTile is { Length: > 0 };
        var vertexStride = includeMaterialFlags ? BaseVertexStride + 1 : BaseVertexStride;
        var vertices = new float[vertexCount * vertexStride];
        var indexCount = (size - 1) * (size - 1) * 6;
        var indices = new uint[indexCount];

        var minHeight = float.MaxValue;
        var maxHeight = float.MinValue;

        for (var i = 0; i < terrain.Height.Length; i++)
        {
            var value = terrain.Height[i];
            if (!float.IsFinite(value))
            {
                continue;
            }
            if (value < minHeight)
            {
                minHeight = value;
            }
            if (value > maxHeight)
            {
                maxHeight = value;
            }
        }

        if (minHeight == float.MaxValue)
        {
            minHeight = 0f;
            maxHeight = 1f;
        }

        for (var y = 0; y < size; y++)
        {
            for (var x = 0; x < size; x++)
            {
                var vertexIndex = y * size + x;
                var offset = vertexIndex * vertexStride;
                var height = SampleHeight(terrain, x, y);

                var position = new Vector3(x * WorldLoader.TerrainScale, height, y * WorldLoader.TerrainScale);
                var normal = ComputeNormal(terrain, x, y);
                var u = x / (float)(size - 1);
                var v = y / (float)(size - 1);

                vertices[offset + 0] = position.X;
                vertices[offset + 1] = position.Y;
                vertices[offset + 2] = position.Z;
                vertices[offset + 3] = normal.X;
                vertices[offset + 4] = normal.Y;
                vertices[offset + 5] = normal.Z;
                vertices[offset + 6] = u;
                vertices[offset + 7] = v;
                if (includeMaterialFlags)
                {
                    var materialIndex = Math.Clamp(vertexIndex, 0, materialFlagsPerTile!.Length - 1);
                    vertices[offset + 8] = materialFlagsPerTile[materialIndex];
                }
            }
        }

        var indexPointer = 0;
        for (var y = 0; y < size - 1; y++)
        {
            for (var x = 0; x < size - 1; x++)
            {
                var topLeft = (uint)(y * size + x);
                var topRight = (uint)(y * size + x + 1);
                var bottomLeft = (uint)((y + 1) * size + x);
                var bottomRight = (uint)((y + 1) * size + x + 1);

                indices[indexPointer++] = topLeft;
                indices[indexPointer++] = bottomLeft;
                indices[indexPointer++] = topRight;
                indices[indexPointer++] = topRight;
                indices[indexPointer++] = bottomLeft;
                indices[indexPointer++] = bottomRight;
            }
        }

        var boundsMin = new Vector3(0f, minHeight, 0f);
        var extent = (size - 1) * WorldLoader.TerrainScale;
        var boundsMax = new Vector3(extent, maxHeight, extent);

        return new TerrainMesh(vertices, indices, boundsMin, boundsMax, vertexStride);
    }

    private static float SampleHeight(TerrainData terrain, int x, int y)
    {
        var size = WorldLoader.TerrainSize;
        x = Math.Clamp(x, 0, size - 1);
        y = Math.Clamp(y, 0, size - 1);
        var value = terrain.Height[y * size + x];
        if (!float.IsFinite(value))
        {
            return 0f;
        }
        return value;
    }

    private static Vector3 ComputeNormal(TerrainData terrain, int x, int y)
    {
        var hL = SampleHeight(terrain, x - 1, y);
        var hR = SampleHeight(terrain, x + 1, y);
        var hD = SampleHeight(terrain, x, y + 1);
        var hU = SampleHeight(terrain, x, y - 1);

        var scale = WorldLoader.TerrainScale;
        var dx = (hR - hL) / (2f * scale);
        var dz = (hD - hU) / (2f * scale);
        var normal = new Vector3(-dx, 1f, -dz);
        if (normal.LengthSquared() <= float.Epsilon)
        {
            return Vector3.UnitY;
        }
        return Vector3.Normalize(normal);
    }
}
