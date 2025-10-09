using System.Buffers.Binary;
using System.Globalization;
using System.Linq;
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
    public short[] BoneIndices { get; init; } = Array.Empty<short>();
}

public sealed class BmdAction
{
    public ushort KeyframeCount { get; init; }
    public bool LockPositions { get; init; }
    public IReadOnlyList<Vector3> LockedPositions { get; init; } = Array.Empty<Vector3>();
}

public sealed class BmdBoneAnimation
{
    public IReadOnlyList<Vector3> Positions { get; init; } = Array.Empty<Vector3>();
    public IReadOnlyList<Vector3> Rotations { get; init; } = Array.Empty<Vector3>();
    public IReadOnlyList<Quaternion> Quaternions { get; init; } = Array.Empty<Quaternion>();
}

public sealed class BmdBone
{
    public string Name { get; init; } = string.Empty;
    public short Parent { get; init; }
    public bool IsDummy { get; init; }
    public Vector3 RestTranslation { get; init; }
    public Vector3 RestRotation { get; init; }
    public Quaternion RestQuaternion { get; init; }
    public IReadOnlyList<BmdBoneAnimation> Animations { get; init; } = Array.Empty<BmdBoneAnimation>();
}

public sealed class BmdModel
{
    public string Name { get; init; } = string.Empty;
    public int Version { get; init; }
    public IReadOnlyList<BmdMesh> Meshes { get; init; } = Array.Empty<BmdMesh>();
    public IReadOnlyList<BmdAction> Actions { get; init; } = Array.Empty<BmdAction>();
    public IReadOnlyList<BmdBone> Bones { get; init; } = Array.Empty<BmdBone>();
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
            EnsureAvailable(data, ptr, 4, "Cabeçalho BMD truncado");
            var size = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(ptr));
            ptr += 4;
            EnsureAvailable(data, ptr, size, "Bloco criptografado incompleto");
            data = MapCrypto.Decrypt(data.AsSpan(ptr, size));
            ptr = 0;
        }

        var name = ReadCString(data, ref ptr, 32);
        EnsureAvailable(data, ptr, 6, "BMD incompleto");
        var numMesh = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
        var numBones = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
        var numActions = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;

        var meshes = new List<BmdMesh>(numMesh);
        for (var meshIndex = 0; meshIndex < numMesh; meshIndex++)
        {
            EnsureAvailable(data, ptr, 10, "Cabeçalho de malha truncado");
            var numVertices = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
            var numNormals = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
            var numTexCoords = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
            var numTriangles = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr)); ptr += 2;
            ptr += 2; // índice de textura (não utilizado diretamente)

            var vertices = new Vector3[numVertices];
            var vertexBones = new short[numVertices];
            for (var i = 0; i < numVertices; i++)
            {
                EnsureAvailable(data, ptr, 16, "Dados de vértice truncados");
                var node = BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(ptr));
                ptr += 2;
                ptr += 2; // alinhamento
                var x = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                var y = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                var z = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                vertexBones[i] = node;
                vertices[i] = new Vector3(x, y, z);
            }

            var normalsSkip = numNormals * 20;
            EnsureAvailable(data, ptr, normalsSkip, "Normais truncadas");
            ptr += normalsSkip;

            var texCoords = new Vector2[numTexCoords];
            for (var i = 0; i < numTexCoords; i++)
            {
                EnsureAvailable(data, ptr, 8, "Coordenadas UV truncadas");
                var u = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                var v = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                texCoords[i] = new Vector2(u, 1.0f - v);
            }

            var meshVertices = new List<Vector3>();
            var meshNormals = new List<Vector3>();
            var meshUvs = new List<Vector2>();
            var meshIndices = new List<uint>();
            var meshBoneRefs = new List<short>();

            for (var triangleIndex = 0; triangleIndex < numTriangles; triangleIndex++)
            {
                EnsureAvailable(data, ptr, 62, "Triângulos truncados");
                var polygon = (sbyte)data[ptr++];
                ptr++; // alinhamento para short

                var vertexIdx = new short[4];
                for (var j = 0; j < 4; j++)
                {
                    vertexIdx[j] = BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(ptr));
                    ptr += 2;
                }

                ptr += 8; // índices de normais não utilizados

                var texIdx = new short[4];
                for (var j = 0; j < 4; j++)
                {
                    texIdx[j] = BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(ptr));
                    ptr += 2;
                }

                ptr += 32; // coordenadas de lightmap
                ptr += 2; // índice de lightmap
                ptr += 2; // padding

                void AppendTriangle(ReadOnlySpan<int> order)
                {
                    var pts = new (Vector3 Position, Vector2 UV, short Bone)[order.Length];
                    for (var k = 0; k < order.Length; k++)
                    {
                        var vid = vertexIdx[order[k]];
                        var tid = texIdx[order[k]];
                        var position = SafeFetch(vertices, vid);
                        var uv = SafeFetch(texCoords, tid);
                        var bone = (short)SafeFetch(vertexBones, vid);
                        pts[k] = (position, uv, bone);
                    }
                    var v0 = pts[0].Position;
                    var v1 = pts[1].Position;
                    var v2 = pts[2].Position;
                    var normal = Vector3.Normalize(Vector3.Cross(v1 - v0, v2 - v0));
                    if (!float.IsFinite(normal.X))
                    {
                        normal = Vector3.UnitZ;
                    }
                    foreach (var pt in pts)
                    {
                        meshVertices.Add(pt.Position);
                        meshUvs.Add(pt.UV);
                        meshNormals.Add(normal);
                        meshBoneRefs.Add(pt.Bone);
                        meshIndices.Add((uint)(meshVertices.Count - 1));
                    }
                }

                if (polygon == 4)
                {
                    AppendTriangle(stackalloc int[] { 0, 1, 2 });
                    AppendTriangle(stackalloc int[] { 0, 2, 3 });
                }
                else
                {
                    AppendTriangle(stackalloc int[] { 0, 1, 2 });
                }
            }

            var textureName = ReadCString(data, ref ptr, 32);
            var mesh = new BmdMesh
            {
                Name = string.IsNullOrEmpty(textureName) ? $"mesh{meshIndex}" : textureName,
                Positions = Flatten(meshVertices),
                Normals = Flatten(meshNormals),
                TexCoords = Flatten(meshUvs),
                Indices = meshIndices.ToArray(),
                TextureName = textureName,
                MaterialFlags = materials.Lookup(textureName).ToFlags(),
                BoneIndices = meshBoneRefs.ToArray(),
            };
            meshes.Add(mesh);
        }

        var actions = new List<BmdAction>(numActions);
        for (var actionIndex = 0; actionIndex < numActions; actionIndex++)
        {
            EnsureAvailable(data, ptr, 3, "Cabeçalho de animação truncado");
            var keyframes = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(ptr));
            ptr += 2;
            var lockPositions = data[ptr++] != 0;
            var lockedPositions = Array.Empty<Vector3>();
            if (lockPositions && keyframes > 0)
            {
                lockedPositions = new Vector3[keyframes];
                for (var frame = 0; frame < keyframes; frame++)
                {
                    EnsureAvailable(data, ptr, 12, "Posições de ação truncadas");
                    var x = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                    var y = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                    var z = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                    lockedPositions[frame] = new Vector3(x, y, z);
                }
            }

            actions.Add(new BmdAction
            {
                KeyframeCount = keyframes,
                LockPositions = lockPositions,
                LockedPositions = lockedPositions,
            });
        }

        var bones = new List<BmdBone>(numBones);
        var channelStride = actions.Sum(action => (int)action.KeyframeCount) * 24;

        for (var boneIndex = 0; boneIndex < numBones; boneIndex++)
        {
            EnsureAvailable(data, ptr, 1, "Dados de osso truncados");
            var dummy = data[ptr++] != 0;
            if (dummy)
            {
                bones.Add(new BmdBone
                {
                    IsDummy = true,
                    Parent = -1,
                });
                continue;
            }

            var boneName = ReadCString(data, ref ptr, 32);
            EnsureAvailable(data, ptr, 2, "Dados de osso truncados");
            var parent = BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(ptr));
            ptr += 2;

            var restTranslation = Vector3.Zero;
            var restRotation = Vector3.Zero;
            var restQuaternion = Quaternion.Identity;

            var remainingBones = numBones - boneIndex;
            var bytesRemaining = data.Length - ptr;
            var includeRest = channelStride > 0
                ? bytesRemaining >= remainingBones * (channelStride + 24)
                : bytesRemaining >= remainingBones * 24;

            if (includeRest && ptr + 24 <= data.Length)
            {
                var tx = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                var ty = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                var tz = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                restTranslation = new Vector3(tx, ty, tz);

                var rx = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                var ry = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                var rz = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                restRotation = new Vector3(rx, ry, rz);
                restQuaternion = EulerToQuaternion(restRotation);
            }

            var boneAnimations = new List<BmdBoneAnimation>(actions.Count);
            foreach (var action in actions)
            {
                var keyframes = action.KeyframeCount;
                var positions = new Vector3[keyframes];
                var rotations = new Vector3[keyframes];
                var quaternions = new Quaternion[keyframes];

                for (var frame = 0; frame < keyframes; frame++)
                {
                    EnsureAvailable(data, ptr, 12, "Posições de osso truncadas");
                    var x = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                    var y = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                    var z = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                    positions[frame] = new Vector3(x, y, z);
                }

                for (var frame = 0; frame < keyframes; frame++)
                {
                    EnsureAvailable(data, ptr, 12, "Rotações de osso truncadas");
                    var rx = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                    var ry = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                    var rz = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(ptr)); ptr += 4;
                    var angles = new Vector3(rx, ry, rz);
                    rotations[frame] = angles;
                    quaternions[frame] = EulerToQuaternion(angles);
                }

                boneAnimations.Add(new BmdBoneAnimation
                {
                    Positions = positions,
                    Rotations = rotations,
                    Quaternions = quaternions,
                });
            }

            bones.Add(new BmdBone
            {
                Name = boneName,
                Parent = parent,
                IsDummy = false,
                RestTranslation = restTranslation,
                RestRotation = restRotation,
                RestQuaternion = restQuaternion,
                Animations = boneAnimations,
            });
        }

        return new BmdModel
        {
            Name = string.IsNullOrEmpty(name) ? Path.GetFileNameWithoutExtension(path) : name,
            Version = version,
            Meshes = meshes,
            Actions = actions,
            Bones = bones,
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

    private static void EnsureAvailable(byte[] data, int offset, int required, string message)
    {
        if (offset < 0 || required < 0 || offset + required > data.Length)
        {
            throw new InvalidDataException(message);
        }
    }

    private static Quaternion EulerToQuaternion(Vector3 angles)
    {
        var halfZ = angles.Z * 0.5f;
        var halfY = angles.Y * 0.5f;
        var halfX = angles.X * 0.5f;

        var sy = MathF.Sin(halfZ);
        var cy = MathF.Cos(halfZ);
        var sp = MathF.Sin(halfY);
        var cp = MathF.Cos(halfY);
        var sr = MathF.Sin(halfX);
        var cr = MathF.Cos(halfX);

        var x = sr * cp * cy - cr * sp * sy;
        var y = cr * sp * cy + sr * cp * sy;
        var z = cr * cp * sy - sr * sp * cy;
        var w = cr * cp * cy + sr * sp * sy;
        return new Quaternion(x, y, z, w);
    }
}
