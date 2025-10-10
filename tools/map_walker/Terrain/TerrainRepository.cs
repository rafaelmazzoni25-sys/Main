using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using MapWalker.Utilities;

namespace MapWalker.Terrain;

internal enum AttributeVariant
{
    Auto,
    Base,
    Event1,
    Event2,
}

internal sealed class TerrainRepository
{
    private static readonly Dictionary<int, string> MapNames = new()
    {
        { 0, "Lorencia" },
        { 1, "Dungeon" },
        { 2, "Devias" },
        { 3, "Noria" },
        { 4, "Lost Tower" },
        { 5, "Exile" },
        { 6, "Arena" },
        { 7, "Atlans" },
        { 8, "Tarkan" },
        { 9, "Devil Square" },
        { 10, "Icarus" },
        { 11, "Blood Castle 1" },
        { 12, "Blood Castle 2" },
        { 13, "Blood Castle 3" },
        { 14, "Blood Castle 4" },
        { 15, "Blood Castle 5" },
        { 16, "Blood Castle 6" },
        { 17, "Blood Castle 7" },
        { 18, "Chaos Castle 1" },
        { 19, "Chaos Castle 2" },
        { 20, "Chaos Castle 3" },
        { 21, "Chaos Castle 4" },
        { 22, "Chaos Castle 5" },
        { 23, "Chaos Castle 6" },
        { 24, "Kalima 1" },
        { 25, "Kalima 2" },
        { 26, "Kalima 3" },
        { 27, "Kalima 4" },
        { 28, "Kalima 5" },
        { 29, "Kalima 6" },
        { 30, "Battle Castle" },
        { 31, "Hunting Ground" },
        { 33, "Aida" },
        { 34, "Crywolf (Campo)" },
        { 35, "Crywolf (Fortaleza)" },
        { 37, "Kanturu 1" },
        { 38, "Kanturu 2" },
        { 39, "Kanturu 3" },
        { 40, "GM Area" },
        { 41, "Changeup 3rd (1)" },
        { 42, "Changeup 3rd (2)" },
        { 45, "Cursed Temple 1" },
        { 46, "Cursed Temple 2" },
        { 47, "Cursed Temple 3" },
        { 48, "Cursed Temple 4" },
        { 49, "Cursed Temple 5" },
        { 50, "Cursed Temple 6" },
        { 51, "Home (6º personagem)" },
        { 52, "Blood Castle Master" },
        { 53, "Chaos Castle Master" },
        { 54, "Character Scene" },
        { 55, "Login Scene" },
        { 56, "Swamp of Calmness" },
        { 57, "Raklion" },
        { 58, "Raklion Boss" },
        { 62, "Santa Town" },
        { 63, "PK Field" },
        { 64, "Duel Arena" },
        { 65, "Doppelganger 1" },
        { 66, "Doppelganger 2" },
        { 67, "Doppelganger 3" },
        { 68, "Doppelganger 4" },
        { 69, "Empire Guardian 1" },
        { 70, "Empire Guardian 2" },
        { 71, "Empire Guardian 3" },
        { 72, "Empire Guardian 4" },
        { 73, "New Login Scene" },
        { 74, "New Character Scene" },
        { 77, "New Login Scene (Alt)" },
        { 78, "New Character Scene (Alt)" },
        { 79, "Lorencia Market" },
        { 80, "Karutan 1" },
        { 81, "Karutan 2" },
    };

    private static readonly string[][] BaseTileFiles =
    {
        new[] { "TileGrass01.jpg", "TileGrass01.tga" },
        new[] { "TileGrass02.jpg", "TileGrass02.tga" },
        new[] { "AlphaTileGround01.Tga", "TileGround01.jpg" },
        new[] { "AlphaTileGround02.Tga", "TileGround02.jpg" },
        new[] { "AlphaTileGround03.Tga", "TileGround03.jpg" },
        new[] { "TileWater01.jpg" },
        new[] { "TileWood01.jpg" },
        new[] { "TileRock01.jpg" },
        new[] { "TileRock02.jpg" },
        new[] { "TileRock03.jpg" },
        new[] { "AlphaTile01.Tga", "TileRock04.jpg" },
        new[] { "Object64/song_lava1.jpg", "TileRock05.jpg" },
        new[] { "AlphaTile01.Tga", "TileRock06.jpg" },
        new[] { "TileRock07.jpg" },
    };

    private static readonly string[] ExtTilePatterns =
    {
        "ExtTile{0:00}.jpg",
        "ExtTile{0}.jpg",
        "ExtTile{0:00}.tga",
    };

    private DirectoryInfo? _dataRoot;

    public DirectoryInfo? DataRoot => _dataRoot;

    public DirectoryInfo? SetDataRoot(string? path)
    {
        var resolved = ResolveDataRoot(path);
        _dataRoot = resolved;
        return resolved;
    }

    public static DirectoryInfo? ResolveDataRoot(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return null;
        }

        var candidate = new DirectoryInfo(path);
        if (!candidate.Exists)
        {
            return null;
        }

        if (candidate.EnumerateDirectories("World1").Any())
        {
            return candidate;
        }

        var nested = new DirectoryInfo(Path.Combine(candidate.FullName, "Data"));
        if (nested.Exists && nested.EnumerateDirectories("World1").Any())
        {
            return nested;
        }

        return null;
    }

    public IReadOnlyList<MapDescriptor> ListMaps()
    {
        if (_dataRoot is null)
        {
            return Array.Empty<MapDescriptor>();
        }

        var descriptors = new List<MapDescriptor>();
        foreach (var folder in _dataRoot.EnumerateDirectories("World*", SearchOption.TopDirectoryOnly).OrderBy(d => d.Name, StringComparer.OrdinalIgnoreCase))
        {
            var digits = new string(folder.Name.Where(char.IsDigit).ToArray());
            if (digits.Length == 0 || !int.TryParse(digits, NumberStyles.Integer, CultureInfo.InvariantCulture, out var worldIndex))
            {
                continue;
            }

            var mapIndex = worldIndex - 1;
            var name = MapNames.TryGetValue(mapIndex, out var display) ? display : $"World {mapIndex}";
            descriptors.Add(new MapDescriptor(mapIndex, worldIndex, name, folder));
        }

        return descriptors;
    }

    public TerrainData LoadMap(MapDescriptor descriptor, AttributeVariant variant)
    {
        if (_dataRoot is null)
        {
            throw new InvalidOperationException("Pasta de dados não configurada.");
        }

        var heights = LoadHeights(descriptor);
        var attributes = LoadAttributes(descriptor, variant);
        var objects = LoadObjects(descriptor);
        var (layer1, layer2, alpha) = LoadTileMapping(descriptor);
        var tileImages = LoadTileTextures(descriptor);
        return new TerrainData(descriptor, heights, attributes, objects, layer1, layer2, alpha, tileImages);
    }

    private static float[,] LoadHeights(MapDescriptor descriptor)
    {
        var worldPath = descriptor.WorldDirectory.FullName;
        var extended = Path.Combine(worldPath, "TerrainHeightNew.OZB");
        if (File.Exists(extended))
        {
            return ReadExtendedHeight(extended);
        }

        var classic = Path.Combine(worldPath, "TerrainHeight.OZB");
        if (File.Exists(classic))
        {
            var scale = descriptor.MapIndex is 55 or 73 or 77 ? 3f : 1.5f;
            return ReadClassicHeight(classic, scale);
        }

        var bmp = Path.Combine(worldPath, "TerrainHeight.bmp");
        if (File.Exists(bmp))
        {
            return ReadBitmapHeight(bmp);
        }

        throw new FileNotFoundException($"Não foram encontrados arquivos de altura em {worldPath}.");
    }

    private static float[,] ReadClassicHeight(string path, float scale)
    {
        var data = File.ReadAllBytes(path);
        const int header = 4 + 1080;
        var expected = header + TerrainData.TerrainSize * TerrainData.TerrainSize;
        if (data.Length < expected)
        {
            throw new InvalidDataException($"Arquivo {path} é muito pequeno para o formato clássico.");
        }

        var payload = new ReadOnlySpan<byte>(data, header, TerrainData.TerrainSize * TerrainData.TerrainSize);
        var result = new float[TerrainData.TerrainSize, TerrainData.TerrainSize];
        var index = 0;
        for (var z = 0; z < TerrainData.TerrainSize; z++)
        {
            for (var x = 0; x < TerrainData.TerrainSize; x++)
            {
                result[z, x] = payload[index++] * scale;
            }
        }

        return result;
    }

    private static float[,] ReadExtendedHeight(string path)
    {
        var data = File.ReadAllBytes(path);
        const int header = 4 + 54;
        var expected = header + TerrainData.TerrainSize * TerrainData.TerrainSize * 3;
        if (data.Length < expected)
        {
            throw new InvalidDataException($"Arquivo {path} é muito pequeno para o formato estendido.");
        }

        var payload = new ReadOnlySpan<byte>(data, header, TerrainData.TerrainSize * TerrainData.TerrainSize * 3);
        var result = new float[TerrainData.TerrainSize, TerrainData.TerrainSize];
        var index = 0;
        for (var z = 0; z < TerrainData.TerrainSize; z++)
        {
            for (var x = 0; x < TerrainData.TerrainSize; x++)
            {
                var b = payload[index++];
                var g = payload[index++];
                var r = payload[index++];
                var value = (r | (g << 8) | (b << 16)) + TerrainData.MinExtendedHeight;
                result[z, x] = value;
            }
        }

        return result;
    }

    private static float[,] ReadBitmapHeight(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        var header = reader.ReadBytes(54);
        if (header.Length != 54)
        {
            throw new InvalidDataException($"Cabeçalho inválido em {path}.");
        }

        var width = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(18, 4));
        var height = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(22, 4));
        if (width != TerrainData.TerrainSize || height != TerrainData.TerrainSize)
        {
            throw new InvalidDataException($"Dimensão inesperada no BMP de terreno: {width}x{height}.");
        }

        var payload = reader.ReadBytes(TerrainData.TerrainSize * TerrainData.TerrainSize);
        if (payload.Length != TerrainData.TerrainSize * TerrainData.TerrainSize)
        {
            throw new InvalidDataException("BMP de terreno truncado.");
        }

        var result = new float[TerrainData.TerrainSize, TerrainData.TerrainSize];
        var index = 0;
        for (var z = 0; z < TerrainData.TerrainSize; z++)
        {
            for (var x = 0; x < TerrainData.TerrainSize; x++)
            {
                result[z, x] = payload[index++] * 1.5f;
            }
        }

        return result;
    }

    private static byte[,]? LoadAttributes(MapDescriptor descriptor, AttributeVariant variant)
    {
        foreach (var candidate in AttributeCandidates(descriptor, variant))
        {
            var name = $"EncTerrain{candidate}.att";
            var path = Path.Combine(descriptor.WorldDirectory.FullName, name);
            if (!File.Exists(path))
            {
                continue;
            }

            return ReadAttribute(path);
        }

        return null;
    }

    private static IEnumerable<int> AttributeCandidates(MapDescriptor descriptor, AttributeVariant variant)
    {
        var baseIndex = descriptor.WorldIndex;
        return variant switch
        {
            AttributeVariant.Base => new[] { baseIndex },
            AttributeVariant.Event1 => new[] { baseIndex * 10 + 1 },
            AttributeVariant.Event2 => new[] { baseIndex * 10 + 2 },
            _ => new[] { baseIndex, baseIndex * 10 + 1, baseIndex * 10 + 2 },
        };
    }

    private static byte[,] ReadAttribute(string path)
    {
        var raw = File.ReadAllBytes(path);
        var decrypted = MapFileDecrypt(raw);
        if (decrypted.Length < 4)
        {
            throw new InvalidDataException("Arquivo de atributo inválido.");
        }

        var count = BinaryPrimitives.ReadUInt16LittleEndian(decrypted.AsSpan(0, 2));
        if (count < TerrainData.TerrainSize * TerrainData.TerrainSize)
        {
            throw new InvalidDataException("Atributos truncados.");
        }

        var buffer = decrypted.AsSpan(4, TerrainData.TerrainSize * TerrainData.TerrainSize).ToArray();
        BuxConvert(buffer);
        var result = new byte[TerrainData.TerrainSize, TerrainData.TerrainSize];
        var index = 0;
        for (var z = 0; z < TerrainData.TerrainSize; z++)
        {
            for (var x = 0; x < TerrainData.TerrainSize; x++)
            {
                result[z, x] = buffer[index++];
            }
        }

        return result;
    }

    private static IReadOnlyList<ObjectInstance> LoadObjects(MapDescriptor descriptor)
    {
        var path = Path.Combine(descriptor.WorldDirectory.FullName, $"EncTerrain{descriptor.WorldIndex}.obj");
        if (!File.Exists(path))
        {
            return Array.Empty<ObjectInstance>();
        }

        var raw = File.ReadAllBytes(path);
        var data = MapFileDecrypt(raw);
        if (data.Length < 4)
        {
            return Array.Empty<ObjectInstance>();
        }

        var count = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(2, 2));
        var objects = new List<ObjectInstance>((int)count);
        var offset = 4;
        for (var i = 0; i < count; i++)
        {
            if (offset + 2 + 12 + 12 + 4 > data.Length)
            {
                break;
            }

            var type = (short)BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(offset, 2));
            offset += 2;
            var px = BitConverter.ToSingle(data, offset);
            var py = BitConverter.ToSingle(data, offset + 4);
            var pz = BitConverter.ToSingle(data, offset + 8);
            offset += 12;
            var rx = BitConverter.ToSingle(data, offset);
            var ry = BitConverter.ToSingle(data, offset + 4);
            var rz = BitConverter.ToSingle(data, offset + 8);
            offset += 12;
            var scale = BitConverter.ToSingle(data, offset);
            offset += 4;
            objects.Add(new ObjectInstance(type, new(px, py, pz), new(rx, ry, rz), scale));
        }

        return objects;
    }

    private static (byte[,]? Layer1, byte[,]? Layer2, float[,]? Alpha) LoadTileMapping(MapDescriptor descriptor)
    {
        var worldPath = descriptor.WorldDirectory.FullName;
        var candidates = new[]
        {
            Path.Combine(worldPath, $"EncTerrain{descriptor.WorldIndex}.map"),
            Path.Combine(worldPath, $"EncTerraintest{descriptor.WorldIndex}.map"),
        };

        foreach (var candidate in candidates)
        {
            if (!File.Exists(candidate))
            {
                continue;
            }

            return ReadTileMapping(candidate);
        }

        return (null, null, null);
    }

    private static (byte[,] Layer1, byte[,] Layer2, float[,] Alpha) ReadTileMapping(string path)
    {
        var raw = File.ReadAllBytes(path);
        var data = MapFileDecrypt(raw);
        var expected = 2 + TerrainData.TerrainSize * TerrainData.TerrainSize * 3;
        if (data.Length < expected)
        {
            throw new InvalidDataException($"Arquivo de mapeamento muito pequeno: {path}");
        }

        var offset = 2;
        var size = TerrainData.TerrainSize * TerrainData.TerrainSize;
        var layer1 = data.AsSpan(offset, size).ToArray();
        offset += size;
        var layer2 = data.AsSpan(offset, size).ToArray();
        offset += size;
        var alphaRaw = data.AsSpan(offset, size).ToArray();
        var layer1Grid = new byte[TerrainData.TerrainSize, TerrainData.TerrainSize];
        var layer2Grid = new byte[TerrainData.TerrainSize, TerrainData.TerrainSize];
        var alphaGrid = new float[TerrainData.TerrainSize, TerrainData.TerrainSize];
        var index = 0;
        for (var z = 0; z < TerrainData.TerrainSize; z++)
        {
            for (var x = 0; x < TerrainData.TerrainSize; x++)
            {
                layer1Grid[z, x] = layer1[index];
                layer2Grid[z, x] = layer2[index];
                alphaGrid[z, x] = alphaRaw[index] / 255f;
                index++;
            }
        }

        return (layer1Grid, layer2Grid, alphaGrid);
    }

    private IReadOnlyList<TileImage> LoadTileTextures(MapDescriptor descriptor)
    {
        if (_dataRoot is null)
        {
            return Array.Empty<TileImage>();
        }

        var images = new List<TileImage>();
        foreach (var candidates in BaseTileFiles)
        {
            var image = LoadFirstTileImage(descriptor, candidates);
            images.Add(image ?? PlaceholderTile());
        }

        for (var i = 1; i <= 16; i++)
        {
            var names = ExtTilePatterns.Select(p => string.Format(CultureInfo.InvariantCulture, p, i)).ToArray();
            var image = LoadFirstTileImage(descriptor, names);
            images.Add(image ?? PlaceholderTile());
        }

        return images;
    }

    private TileImage? LoadFirstTileImage(MapDescriptor descriptor, IEnumerable<string> candidates)
    {
        foreach (var name in candidates)
        {
            var image = LoadTileImage(descriptor, name);
            if (image is not null)
            {
                return image;
            }
        }

        return null;
    }

    private TileImage? LoadTileImage(MapDescriptor descriptor, string relative)
    {
        var relativePath = relative.Replace('/', Path.DirectorySeparatorChar);
        var search = new List<string>
        {
            Path.Combine(descriptor.WorldDirectory.FullName, relativePath)
        };

        if (_dataRoot is not null)
        {
            search.Add(Path.Combine(_dataRoot.FullName, relativePath));
        }

        foreach (var candidate in search)
        {
            if (!File.Exists(candidate))
            {
                continue;
            }

            try
            {
                var file = new FileInfo(candidate);
                return ImageLoader.LoadImage(file);
            }
            catch
            {
                // ignore and try next
            }
        }

        return null;
    }

    private static TileImage PlaceholderTile()
    {
        var pixels = new byte[4 * 4 * 4];
        for (var i = 0; i < pixels.Length; i += 4)
        {
            pixels[i + 0] = 96;
            pixels[i + 1] = 96;
            pixels[i + 2] = 96;
            pixels[i + 3] = 255;
        }

        return new TileImage(4, 4, pixels);
    }

    public static byte[] MapFileDecrypt(ReadOnlySpan<byte> data)
    {
        var key = new byte[]
        {
            0xD1, 0x73, 0x52, 0xF6, 0xD2, 0x9A, 0xCB, 0x27,
            0x3E, 0xAF, 0x59, 0x31, 0x37, 0xB3, 0xE7, 0xA2
        };

        var result = new byte[data.Length];
        byte wKey = 0x5E;
        for (var i = 0; i < data.Length; i++)
        {
            var value = data[i];
            result[i] = (byte)((value ^ key[i % key.Length]) - wKey);
            wKey = (byte)(value + 0x3D);
        }

        return result;
    }

    private static void BuxConvert(Span<byte> buffer)
    {
        var code = new byte[] { 0xFC, 0xCF, 0xAB };
        for (var i = 0; i < buffer.Length; i++)
        {
            buffer[i] ^= code[i % 3];
        }
    }
}
