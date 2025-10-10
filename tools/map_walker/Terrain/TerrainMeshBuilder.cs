using System;
using OpenTK.Mathematics;

namespace MapWalker.Terrain;

internal static class TerrainMeshBuilder
{
    public static TerrainMesh Build(TerrainData data)
    {
        var heights = data.Heights;
        var size = data.Size;
        var count = size * size;
        var positions = new float[count * 3];
        var normals = new float[count * 3];
        var texCoords = new float[count * 2];
        var tileCoords = new float[count * 2];
        var normalMap = ComputeNormals(heights, size);

        var idx = 0;
        for (var z = 0; z < size; z++)
        {
            for (var x = 0; x < size; x++)
            {
                var baseIndex = idx * 3;
                positions[baseIndex + 0] = x * TerrainData.TerrainScale;
                positions[baseIndex + 1] = heights[z, x];
                positions[baseIndex + 2] = z * TerrainData.TerrainScale;

                var normal = normalMap[z, x];
                normals[baseIndex + 0] = normal.X;
                normals[baseIndex + 1] = normal.Y;
                normals[baseIndex + 2] = normal.Z;

                var uvIndex = idx * 2;
                texCoords[uvIndex + 0] = x * 0.25f;
                texCoords[uvIndex + 1] = z * 0.25f;
                tileCoords[uvIndex + 0] = x;
                tileCoords[uvIndex + 1] = z;
                idx++;
            }
        }

        var indices = new uint[(size - 1) * (size - 1) * 6];
        var face = 0;
        for (var z = 0; z < size - 1; z++)
        {
            for (var x = 0; x < size - 1; x++)
            {
                var i0 = (uint)(z * size + x);
                var i1 = i0 + 1;
                var i2 = (uint)((z + 1) * size + x);
                var i3 = i2 + 1;
                indices[face++] = i0;
                indices[face++] = i2;
                indices[face++] = i1;
                indices[face++] = i1;
                indices[face++] = i2;
                indices[face++] = i3;
            }
        }

        return new TerrainMesh(positions, normals, texCoords, tileCoords, indices);
    }

    private static Vector3[,] ComputeNormals(float[,] heights, int size)
    {
        var result = new Vector3[size, size];
        var scale = TerrainData.TerrainScale;
        for (var z = 0; z < size; z++)
        {
            for (var x = 0; x < size; x++)
            {
                var left = heights[z, Math.Max(x - 1, 0)];
                var right = heights[z, Math.Min(x + 1, size - 1)];
                var up = heights[Math.Max(z - 1, 0), x];
                var down = heights[Math.Min(z + 1, size - 1), x];
                var dx = (right - left) / (2f * scale);
                var dz = (down - up) / (2f * scale);
                var normal = new Vector3(-dx, 1f, -dz);
                if (normal.LengthSquared > 0)
                {
                    normal.Normalize();
                }
                else
                {
                    normal = Vector3.UnitY;
                }

                result[z, x] = normal;
            }
        }

        return result;
    }
}
