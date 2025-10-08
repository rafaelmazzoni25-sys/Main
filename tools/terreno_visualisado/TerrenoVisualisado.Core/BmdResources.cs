using System.Buffers.Binary;
using System.Globalization;
using System.Numerics;
using System.Text;

namespace TerrenoVisualisado.Core;

public sealed class BmdMesh
{
    public string Name { get; init; } = string.Empty;
    public float[] Positions { get; init; } = Array.Empty<float>();
    public float[] Normals { get; init; } = Array.Empty<float>();
    public float[] TexCoords { get; init; } = Array.Empty<float>();
    public uint[] Indices { get; init; } = Array.Empty<uint>();
    public string TextureName { get; init; } = string.Empty;
    public MaterialFlags MaterialFlags { get; init; }
}

public sealed class BmdModel
{
    public string Name { get; init; } = string.Empty;
    public int Version { get; init; }
    public IReadOnlyList<BmdMesh> Meshes { get; init; } = Array.Empty<BmdMesh>();
    public string SourcePath { get; init; } = string.Empty;
}

public sealed class BmdLibrary
{
    private readonly Dictionary<string, List<string>> _index = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, BmdModel> _cache = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _failures = new(StringComparer.OrdinalIgnoreCase);
    private readonly MaterialStateLibrary _materials;

    public BmdLibrary(IEnumerable<string> searchRoots, MaterialStateLibrary materials)
    {
        _materials = materials;
        BuildIndex(searchRoots);
    }

    public IReadOnlyDictionary<string, string> Failures => _failures;

    public BmdModel? Load(ObjectInstance obj)
    {
        var path = Resolve(obj);
        if (path is null)
        {
            return null;
        }
        if (_cache.TryGetValue(path, out var cached))
        {
            return cached;
        }
        try
        {
            var model = BmdLoader.Load(path, _materials);
            _cache[path] = model;
            return model;
        }
        catch (Exception ex)
        {
            _failures[path] = ex.Message;
            return null;
        }
    }

    public string? Resolve(ObjectInstance obj)
    {
        foreach (var key in EnumerateCandidates(obj))
        {
            if (_index.TryGetValue(key, out var paths))
            {
                foreach (var path in paths)
                {
                    if (File.Exists(path))
                    {
                        return path;
                    }
                }
            }
        }
        return null;
    }

    private IEnumerable<string> EnumerateCandidates(ObjectInstance obj)
    {
        if (!string.IsNullOrEmpty(obj.TypeName))
        {
            var lowered = obj.TypeName.ToLowerInvariant();
            yield return lowered;
            if (lowered.StartsWith("model_", StringComparison.OrdinalIgnoreCase))
            {
                var trimmed = lowered[6..];
                yield return trimmed;
                yield return trimmed.Replace("_", string.Empty, StringComparison.Ordinal);
            }
            yield return lowered.Replace("_", string.Empty, StringComparison.Ordinal);
        }
        yield return obj.TypeId.ToString(CultureInfo.InvariantCulture);
        yield return $"object{obj.TypeId}";
        yield return $"object{obj.TypeId:00}";
    }

    private void BuildIndex(IEnumerable<string> searchRoots)
    {
        var unique = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var root in searchRoots)
        {
            if (string.IsNullOrWhiteSpace(root))
            {
                continue;
            }
            var full = Path.GetFullPath(root);
            if (!Directory.Exists(full) || !unique.Add(full))
            {
                continue;
            }
            foreach (var file in Directory.EnumerateFiles(full, "*.bmd", SearchOption.AllDirectories))
            {
                var name = Path.GetFileNameWithoutExtension(file).ToLowerInvariant();
                AddIndex(name, file);
                AddIndex(Path.GetFileName(file).ToLowerInvariant(), file);
                var relative = Path.GetRelativePath(full, file).Replace('\\', '/').ToLowerInvariant();
                AddIndex(relative, file);
                var parent = Path.GetFileName(Path.GetDirectoryName(file) ?? string.Empty).ToLowerInvariant();
                if (parent.StartsWith("object", StringComparison.OrdinalIgnoreCase))
                {
                    AddIndex(parent + "_" + name, file);
                    var digits = new string(name.Where(char.IsDigit).ToArray());
                    if (!string.IsNullOrEmpty(digits))
                    {
                        AddIndex(parent + "_" + digits, file);
                        AddIndex(digits, file);
                    }
                }
            }
        }
    }

    private void AddIndex(string key, string path)
    {
        if (!_index.TryGetValue(key, out var list))
        {
            list = new List<string>();
            _index[key] = list;
        }
        if (!list.Contains(path, StringComparer.OrdinalIgnoreCase))
        {
            list.Add(path);
        }
    }
}

internal static class BmdLoader
{
    public static BmdModel Load(string path, MaterialStateLibrary materials)
    {
        var raw = File.ReadAllBytes(path);
        if (raw.Length < 7 || raw[0] != 'B' || raw[1] != 'M' || raw[2] != 'D')
        {
            throw new InvalidDataException($"Arquivo BMD inválido: {path}");
        }

        var data = raw;
        var ptr = 3;
        byte version = data[ptr++];
        if (version == 12)
        {
            if (ptr + 4 > data.Length)
            {
                throw new InvalidDataException("Cabeçalho BMD truncado");
            }
            var size = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(ptr));
            ptr += 4;
            if (ptr + size > data.Length)
            {
                throw new InvalidDataException("Bloco criptografado incompleto");
            }
            data = MapCrypto.Decrypt(data.AsSpan(ptr, size));
            ptr = 0;
            version = data[ptr++];
        }

        var name = ReadCString(data, ref ptr, 32);
        if (ptr + 6 > data.Length)
        {
            throw new InvalidDataException("BMD incompleto");
        }
        var numMesh = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
        var numBones = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
        var numActions = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;

        var meshes = new List<BmdMesh>();
        for (var meshIndex = 0; meshIndex < numMesh; meshIndex++)
        {
            if (ptr + 10 > data.Length)
            {
                throw new InvalidDataException("Cabeçalho de malha truncado");
            }
            var numVertices = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
            var numNormals = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
            var numTexCoords = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
            var numTriangles = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
            ptr += 2; // texture index (não utilizado)

            var vertices = new Vector3[numVertices];
            var vertexBones = new short[numVertices];
            for (var i = 0; i < numVertices; i++)
            {
                if (ptr + 16 > data.Length)
                {
                    throw new InvalidDataException("Dados de vértice truncados");
                }
                var node = BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(ptr));
                ptr += 2;
                ptr += 2; // padding
                var x = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                var y = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                var z = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                vertexBones[i] = node;
                vertices[i] = new Vector3(x, y, z);
            }

            for (var i = 0; i < numNormals; i++)
            {
                ptr += 18; // ignorar normais originais
            }

            var texCoords = new Vector2[numTexCoords];
            for (var i = 0; i < numTexCoords; i++)
            {
                if (ptr + 8 > data.Length)
                {
                    throw new InvalidDataException("Coordenadas UV truncadas");
                }
                var u = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                var v = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                texCoords[i] = new Vector2(u, 1.0f - v);
            }

            var triangles = new List<(int polygon, short[] vertexIdx, short[] texIdx)>();
            for (var i = 0; i < numTriangles; i++)
            {
                if (ptr + 32 > data.Length)
                {
                    throw new InvalidDataException("Triângulos truncados");
                }
                var polygon = (sbyte)data[ptr++];
                var vertexIdx = new short[4];
                var texIdx = new short[4];
                for (var j = 0; j < 4; j++)
                {
                    vertexIdx[j] = BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
                }
                ptr += 8; // padding
                for (var j = 0; j < 4; j++)
                {
                    texIdx[j] = BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
                }
                ptr += 8; // padding
                triangles.Add((polygon, vertexIdx, texIdx));
            }

            var textureName = ReadCString(data, ref ptr, 32);
            var meshVertices = new List<Vector3>();
            var meshNormals = new List<Vector3>();
            var meshUvs = new List<Vector2>();
            var meshIndices = new List<uint>();

            void AppendTriangle((int polygon, short[] vertexIdx, short[] texIdx) triangle, ReadOnlySpan<int> order)
            {
                var pts = new (Vector3 Position, Vector2 UV, int Bone) [order.Length];
                for (var k = 0; k < order.Length; k++)
                {
                    var vid = triangle.vertexIdx[order[k]];
                    var tid = triangle.texIdx[order[k]];
                    var position = SafeFetch(vertices, vid);
                    var uv = SafeFetch(texCoords, tid);
                    var bone = SafeFetch(vertexBones, vid);
                    pts[k] = (position, uv, bone);
                }
                var v0 = pts[0].Position;
                var v1 = pts[1].Position;
                var v2 = pts[2].Position;
                var normal = Vector3.Normalize(Vector3.Cross(v1 - v0, v2 - v0));
                if (!float.IsFinite(normal.X))
                {
                    normal = Vector3.UnitY;
                }
                foreach (var pt in pts)
                {
                    meshVertices.Add(pt.Position);
                    meshUvs.Add(pt.UV);
                    meshNormals.Add(normal);
                    meshIndices.Add((uint)(meshVertices.Count - 1));
                }
            }

            for (var currentTriangle = 0; currentTriangle < triangles.Count; currentTriangle++)
            {
                var triangle = triangles[currentTriangle];
                var polygon = triangle.polygon;
                if (polygon == 4)
                {
                    AppendTriangle(triangle, stackalloc int[] { 0, 1, 2 });
                    AppendTriangle(triangle, stackalloc int[] { 0, 2, 3 });
                }
                else
                {
                    AppendTriangle(triangle, stackalloc int[] { 0, 1, 2 });
                }
            }

            var mesh = new BmdMesh
            {
                Name = string.IsNullOrEmpty(textureName) ? $"mesh{meshIndex}" : textureName,
                Positions = Flatten(meshVertices),
                Normals = Flatten(meshNormals),
                TexCoords = Flatten(meshUvs),
                Indices = meshIndices.ToArray(),
                TextureName = textureName,
                MaterialFlags = materials.Lookup(textureName).ToFlags(),
            };
            meshes.Add(mesh);
        }

        return new BmdModel
        {
            Name = string.IsNullOrEmpty(name) ? Path.GetFileNameWithoutExtension(path) : name,
            Version = version,
            Meshes = meshes,
            SourcePath = path,
        };
    }

    private static Vector3 SafeFetch(Vector3[] array, int index)
    {
        if (index >= 0 && index < array.Length)
        {
            return array[index];
        }
        return Vector3.Zero;
    }

    private static Vector2 SafeFetch(Vector2[] array, int index)
    {
        if (index >= 0 && index < array.Length)
        {
            return array[index];
        }
        return Vector2.Zero;
    }

    private static int SafeFetch(short[] array, int index)
    {
        if (index >= 0 && index < array.Length)
        {
            return array[index];
        }
        return -1;
    }

    private static float[] Flatten(List<Vector3> values)
    {
        var result = new float[values.Count * 3];
        for (var i = 0; i < values.Count; i++)
        {
            result[i * 3 + 0] = values[i].X;
            result[i * 3 + 1] = values[i].Y;
            result[i * 3 + 2] = values[i].Z;
        }
        return result;
    }

    private static float[] Flatten(List<Vector2> values)
    {
        var result = new float[values.Count * 2];
        for (var i = 0; i < values.Count; i++)
        {
            result[i * 2 + 0] = values[i].X;
            result[i * 2 + 1] = values[i].Y;
        }
        return result;
    }

    private static string ReadCString(byte[] data, ref int offset, int length)
    {
        var span = data.AsSpan(offset, Math.Min(length, data.Length - offset));
        var zero = span.IndexOf((byte)0);
        var actualLength = zero >= 0 ? zero : span.Length;
        var text = Encoding.GetEncoding(1252).GetString(span[..actualLength]);
        offset += length;
        return text.Trim();
    }
}
