using System.Buffers.Binary;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace TerrenoVisualisado;

internal static class Program
{
    private static int Main(string[] args)
    {
        try
        {
            var options = CliOptions.Parse(args);
            if (options.WorldDirectory is null)
            {
                throw new ArgumentException("Informe o diretório --world contendo os arquivos EncTerrain.");
            }

            var loader = new WorldLoader();
            var world = loader.Load(options.WorldDirectory, new WorldLoader.LoadOptions
            {
                ObjectRoot = options.ObjectDirectory,
                MapId = options.MapId,
                EnumPath = options.EnumPath,
                ForceExtendedHeight = options.ForceExtendedHeight,
                HeightScale = options.HeightScale,
            });

            PrintSummary(world);
            SaveJson(world, options.OutputPath ?? "terrainsummary.json");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Erro: {ex.Message}");
            return 1;
        }
    }

    private static void PrintSummary(WorldData world)
    {
        Console.WriteLine($"Diretório: {world.WorldPath}");
        Console.WriteLine($"Mapa detectado: {world.MapId}");
        Console.WriteLine($"Objetos: {world.Objects.Count} (versão {world.ObjectVersion})");

        var tileCounts = new Dictionary<byte, int>();
        for (var i = 0; i < WorldLoader.TerrainSize * WorldLoader.TerrainSize; i++)
        {
            Increment(world.Terrain.Layer1[i]);
            Increment(world.Terrain.Layer2[i]);
        }

        Console.WriteLine("Top tiles por frequência:");
        foreach (var (tile, count) in tileCounts.OrderByDescending(kv => kv.Value).Take(8))
        {
            Console.WriteLine($"  Tile {tile:D3}: {count} ocorrências");
        }

        var attributeCounts = new Dictionary<ushort, int>();
        foreach (var attr in world.Terrain.Attributes)
        {
            attributeCounts.TryGetValue(attr, out var count);
            attributeCounts[attr] = count + 1;
        }
        Console.WriteLine("Atributos mais comuns:");
        foreach (var (attr, count) in attributeCounts.OrderByDescending(kv => kv.Value).Take(8))
        {
            Console.WriteLine($"  0x{attr:X3}: {count} tiles");
        }

        var objectCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var obj in world.Objects)
        {
            var key = obj.TypeName ?? $"ID_{obj.TypeId}";
            objectCounts.TryGetValue(key, out var count);
            objectCounts[key] = count + 1;
        }

        Console.WriteLine("Objetos estáticos mais frequentes:");
        foreach (var (name, count) in objectCounts.OrderByDescending(kv => kv.Value).Take(10))
        {
            Console.WriteLine($"  {name}: {count}");
        }

        void Increment(byte tile)
        {
            tileCounts.TryGetValue(tile, out var count);
            tileCounts[tile] = count + 1;
        }
    }

    private static void SaveJson(WorldData world, string outputPath)
    {
        var export = new
        {
            world.WorldPath,
            world.ObjectsPath,
            world.MapId,
            world.ObjectVersion,
            Terrain = new
            {
                Size = WorldLoader.TerrainSize,
                Height = world.Terrain.Height,
                Layer1 = world.Terrain.Layer1,
                Layer2 = world.Terrain.Layer2,
                Alpha = world.Terrain.Alpha,
                Attributes = world.Terrain.Attributes,
            },
            Objects = world.Objects.Select(obj => new
            {
                obj.TypeId,
                obj.TypeName,
                Position = new[] { obj.Position.X, obj.Position.Y, obj.Position.Z },
                RawPosition = new[] { obj.RawPosition.X, obj.RawPosition.Y, obj.RawPosition.Z },
                Rotation = new[] { obj.Rotation.X, obj.Rotation.Y, obj.Rotation.Z },
                obj.Scale,
            }),
        };

        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
        };
        File.WriteAllText(outputPath, JsonSerializer.Serialize(export, options));
        Console.WriteLine($"Resumo salvo em {outputPath}");
    }

    private sealed record CliOptions
    {
        public string? WorldDirectory { get; init; }
        public string? ObjectDirectory { get; init; }
        public int? MapId { get; init; }
        public string? EnumPath { get; init; }
        public float? HeightScale { get; init; }
        public bool ForceExtendedHeight { get; init; }
        public string? OutputPath { get; init; }

        public static CliOptions Parse(string[] args)
        {
            var options = new CliOptions();
            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];
                switch (arg)
                {
                    case "--world":
                        options.WorldDirectory = RequireValue(args, ref i, arg);
                        break;
                    case "--objects":
                        options.ObjectDirectory = RequireValue(args, ref i, arg);
                        break;
                    case "--map":
                        options.MapId = int.Parse(RequireValue(args, ref i, arg), CultureInfo.InvariantCulture);
                        break;
                    case "--enum":
                        options.EnumPath = RequireValue(args, ref i, arg);
                        break;
                    case "--height-scale":
                        options.HeightScale = float.Parse(RequireValue(args, ref i, arg), CultureInfo.InvariantCulture);
                        break;
                    case "--extended-height":
                        options.ForceExtendedHeight = true;
                        break;
                    case "--output":
                        options.OutputPath = RequireValue(args, ref i, arg);
                        break;
                    default:
                        throw new ArgumentException($"Argumento desconhecido: {arg}");
                }
            }
            return options;
        }

        private static string RequireValue(IReadOnlyList<string> args, ref int index, string name)
        {
            if (index + 1 >= args.Count)
            {
                throw new ArgumentException($"O argumento {name} requer um valor.");
            }
            index++;
            return args[index];
        }
    }
}

internal sealed class WorldLoader
{
    public const int TerrainSize = 256;
    private const float TerrainScale = 100.0f;
    private const float DefaultClassicHeightScale = 1.5f;
    private const float MinHeightBias = -500.0f;

    private static readonly byte[] XorKey =
    {
        0xD1, 0x73, 0x52, 0xF6, 0xD2, 0x9A, 0xCB, 0x27,
        0x3E, 0xAF, 0x59, 0x31, 0x37, 0xB3, 0xE7, 0xA2,
    };

    private static readonly byte[] BuxCode = { 0xFC, 0xCF, 0xAB };

    internal sealed class LoadOptions
    {
        public string? ObjectRoot { get; init; }
        public int? MapId { get; set; }
        public string? EnumPath { get; init; }
        public bool ForceExtendedHeight { get; init; }
        public float? HeightScale { get; init; }
    }

    public WorldData Load(string worldDirectory, LoadOptions options)
    {
        if (!Directory.Exists(worldDirectory))
        {
            throw new DirectoryNotFoundException($"Diretório inválido: {worldDirectory}");
        }

        options.MapId ??= InferMapId(worldDirectory);

        var attributesPath = ResolveTerrainFile(worldDirectory, options.MapId, ".att");
        var mappingPath = ResolveTerrainFile(worldDirectory, options.MapId, ".map");
        var objectsPath = ResolveObjectsPath(worldDirectory, options.ObjectRoot, options.MapId);
        var heightPath = ResolveHeightPath(worldDirectory, options.ForceExtendedHeight);
        var modelNames = LoadModelNames(options.EnumPath);

        var terrain = LoadTerrain(attributesPath, mappingPath, heightPath, options.ForceExtendedHeight, options.HeightScale, out var attributeMapId, out var mappingMapId);
        var objects = LoadObjects(objectsPath, modelNames, out var version, out var objectMapId);

        var resolvedMapId = options.MapId ?? (attributeMapId >= 0 ? attributeMapId : (mappingMapId >= 0 ? mappingMapId : objectMapId));
        AlignObjectsToTerrain(objects, terrain);

        return new WorldData
        {
            WorldPath = Path.GetFullPath(worldDirectory),
            ObjectsPath = Path.GetFullPath(objectsPath),
            MapId = resolvedMapId,
            ObjectVersion = version,
            Terrain = terrain,
            Objects = objects,
        };
    }

    private static void AlignObjectsToTerrain(List<ObjectInstance> objects, TerrainData terrain)
    {
        foreach (var obj in objects)
        {
            var tileX = obj.RawPosition.X / TerrainScale;
            var tileY = obj.RawPosition.Y / TerrainScale;
            var height = BilinearHeight(terrain, tileX, tileY);
            obj.Position = new Vector3(obj.RawPosition.X, height, obj.RawPosition.Y);
        }
    }

    private static float BilinearHeight(TerrainData terrain, float tileX, float tileY)
    {
        if (terrain.Height.Length == 0)
        {
            return 0.0f;
        }

        var maxIndex = TerrainSize - 1;
        var x = Math.Clamp(tileX, 0.0f, maxIndex);
        var y = Math.Clamp(tileY, 0.0f, maxIndex);

        var xi = (int)MathF.Floor(x);
        var yi = (int)MathF.Floor(y);
        var xd = x - xi;
        var yd = y - yi;

        float Sample(int sx, int sy)
        {
            sx = Math.Clamp(sx, 0, maxIndex);
            sy = Math.Clamp(sy, 0, maxIndex);
            return terrain.Height[sy * TerrainSize + sx];
        }

        var h00 = Sample(xi, yi);
        var h10 = Sample(xi + 1, yi);
        var h01 = Sample(xi, yi + 1);
        var h11 = Sample(xi + 1, yi + 1);

        var h0 = h00 * (1 - xd) + h10 * xd;
        var h1 = h01 * (1 - xd) + h11 * xd;
        return h0 * (1 - yd) + h1 * yd;
    }

    private static List<ObjectInstance> LoadObjects(string objectsPath, Dictionary<int, string> modelNames, out int version, out int mapId)
    {
        var raw = File.ReadAllBytes(objectsPath);
        var decrypted = MapFileDecrypt(raw);
        if (decrypted.Length < 4)
        {
            throw new InvalidDataException($"Arquivo EncTerrain.obj truncado: {objectsPath}");
        }

        var span = decrypted.AsSpan();
        var index = 0;
        version = span[index++];
        mapId = span[index++];
        var count = BinaryPrimitives.ReadInt16LittleEndian(span[index..]);
        index += 2;
        if (count < 0)
        {
            throw new InvalidDataException($"Contagem negativa de objetos em {objectsPath}");
        }

        var objects = new List<ObjectInstance>(count);
        for (var i = 0; i < count; i++)
        {
            if (index + 34 > span.Length)
            {
                throw new InvalidDataException($"Dados de objeto insuficientes em {objectsPath}");
            }

            var typeId = BinaryPrimitives.ReadInt16LittleEndian(span[index..]);
            index += 2;
            var position = new Vector3(
                BitConverter.ToSingle(span[index..(index + 4)]),
                BitConverter.ToSingle(span[(index + 4)..(index + 8)]),
                BitConverter.ToSingle(span[(index + 8)..(index + 12)]));
            index += 12;
            var rotation = new Vector3(
                BitConverter.ToSingle(span[index..(index + 4)]),
                BitConverter.ToSingle(span[(index + 4)..(index + 8)]),
                BitConverter.ToSingle(span[(index + 8)..(index + 12)]));
            index += 12;
            var scale = BitConverter.ToSingle(span[index..(index + 4)]);
            index += 4;

            modelNames.TryGetValue(typeId, out var name);
            objects.Add(new ObjectInstance
            {
                TypeId = typeId,
                TypeName = name,
                RawPosition = position,
                Rotation = rotation,
                Scale = scale,
            });
        }

        return objects;
    }

    private static TerrainData LoadTerrain(string attributesPath, string mappingPath, string heightPath, bool forceExtendedHeight, float? heightScale, out int attributeMapId, out int mappingMapId)
    {
        var height = new float[TerrainSize * TerrainSize];
        var layer1 = new byte[TerrainSize * TerrainSize];
        var layer2 = new byte[TerrainSize * TerrainSize];
        var alpha = new float[TerrainSize * TerrainSize];
        var attributes = new ushort[TerrainSize * TerrainSize];

        // atributos
        {
            var raw = File.ReadAllBytes(attributesPath);
            var decrypted = MapFileDecrypt(raw);
            BuxConvert(decrypted);
            if (decrypted.Length != 131076 && decrypted.Length != 65540)
            {
                throw new InvalidDataException($"Tamanho inesperado para arquivo de atributos: {attributesPath}");
            }
            attributeMapId = decrypted[1];
            const int offset = 4;
            if (decrypted.Length == 65540)
            {
                for (var i = 0; i < attributes.Length; i++)
                {
                    attributes[i] = decrypted[offset + i];
                }
            }
            else
            {
                var span = decrypted.AsSpan(offset);
                for (var i = 0; i < attributes.Length; i++)
                {
                    attributes[i] = BinaryPrimitives.ReadUInt16LittleEndian(span.Slice(i * 2, 2));
                }
            }
        }

        // mapping
        {
            var raw = File.ReadAllBytes(mappingPath);
            var decrypted = MapFileDecrypt(raw);
            if (decrypted.Length < 2 + TerrainSize * TerrainSize * 3)
            {
                throw new InvalidDataException($"Arquivo EncTerrain.map truncado: {mappingPath}");
            }
            mappingMapId = decrypted[1];
            var index = 2;
            for (var i = 0; i < layer1.Length; i++)
            {
                layer1[i] = decrypted[index++];
            }
            for (var i = 0; i < layer2.Length; i++)
            {
                layer2[i] = decrypted[index++];
            }
            for (var i = 0; i < alpha.Length; i++)
            {
                alpha[i] = decrypted[index++] / 255.0f;
            }
        }

        // height
        {
            var raw = File.ReadAllBytes(heightPath);
            if (raw.Length < 4)
            {
                throw new InvalidDataException($"Arquivo de altura muito pequeno: {heightPath}");
            }
            var payload = raw.AsSpan(4);
            var extended = forceExtendedHeight || Path.GetFileName(heightPath).Equals("TerrainHeightNew.OZB", StringComparison.OrdinalIgnoreCase);
            if (!extended)
            {
                var expected = 1080 + height.Length;
                if (payload.Length < expected)
                {
                    throw new InvalidDataException($"Arquivo de altura clássico truncado: {heightPath}");
                }
                var scale = heightScale ?? DefaultClassicHeightScale;
                var heightBytes = payload.Slice(1080);
                for (var i = 0; i < height.Length; i++)
                {
                    height[i] = heightBytes[i] * scale;
                }
            }
            else
            {
                const int headerSize = 14 + 40;
                var expected = headerSize + height.Length * 3;
                if (payload.Length < expected)
                {
                    throw new InvalidDataException($"Arquivo TerrainHeightNew.OZB truncado: {heightPath}");
                }
                var pixel = payload.Slice(headerSize);
                for (var i = 0; i < height.Length; i++)
                {
                    var b = pixel[i * 3 + 0];
                    var g = pixel[i * 3 + 1];
                    var r = pixel[i * 3 + 2];
                    var value = (r << 16) | (g << 8) | b;
                    height[i] = value + MinHeightBias;
                }
            }
        }

        return new TerrainData(height, layer1, layer2, alpha, attributes);
    }

    private static Dictionary<int, string> LoadModelNames(string? enumPath)
    {
        var result = new Dictionary<int, string>();
        if (string.IsNullOrEmpty(enumPath) || !File.Exists(enumPath))
        {
            return result;
        }

        var regex = new Regex("^\\s*(MODEL_[A-Z0-9_]+)\\s*=\\s*([^,]+)", RegexOptions.Compiled);
        foreach (var line in File.ReadLines(enumPath))
        {
            var match = regex.Match(line);
            if (!match.Success)
            {
                continue;
            }

            var name = match.Groups[1].Value.Trim();
            var valueText = match.Groups[2].Value.Trim();
            if (valueText.EndsWith(",", StringComparison.Ordinal))
            {
                valueText = valueText[..^1];
            }

            if (!TryParseInt(valueText, out var value))
            {
                continue;
            }
            result[value] = name;
        }
        return result;
    }

    private static bool TryParseInt(string text, out int value)
    {
        text = text.Trim();
        if (text.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
        {
            return int.TryParse(text[2..], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out value);
        }
        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
    }

    private static string ResolveObjectsPath(string worldDirectory, string? objectRoot, int? mapId)
    {
        var candidates = new List<string>();
        if (!string.IsNullOrEmpty(objectRoot))
        {
            candidates.Add(objectRoot);
        }
        if (GuessObjectFolder(worldDirectory) is { } guessed)
        {
            candidates.Add(guessed);
        }
        candidates.Add(worldDirectory);

        foreach (var candidate in candidates)
        {
            try
            {
                return ResolveTerrainFile(candidate, mapId, ".obj");
            }
            catch
            {
                // tenta próxima opção
            }
        }

        throw new FileNotFoundException("Arquivo EncTerrain*.obj não encontrado. Utilize --objects para informar a pasta correta.");
    }

    private static string ResolveHeightPath(string worldDirectory, bool preferExtended)
    {
        var classic = Path.Combine(worldDirectory, "TerrainHeight.OZB");
        var extended = Path.Combine(worldDirectory, "TerrainHeightNew.OZB");

        bool classicValid = File.Exists(classic) && new FileInfo(classic).Length >= 4 + 1080 + TerrainSize * TerrainSize;
        bool extendedValid = File.Exists(extended) && new FileInfo(extended).Length >= 4 + 54 + TerrainSize * TerrainSize * 3;

        if (preferExtended && extendedValid)
        {
            return extended;
        }
        if (classicValid)
        {
            return classic;
        }
        if (extendedValid)
        {
            return extended;
        }
        throw new FileNotFoundException("Arquivos TerrainHeight.OZB ou TerrainHeightNew.OZB não encontrados ou inválidos.");
    }

    private static string ResolveTerrainFile(string directory, int? mapId, string extension)
    {
        if (!Directory.Exists(directory))
        {
            throw new DirectoryNotFoundException($"Diretório inválido: {directory}");
        }

        var matches = Directory.EnumerateFiles(directory, "EncTerrain*" + extension, SearchOption.TopDirectoryOnly)
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (matches.Count == 0)
        {
            throw new FileNotFoundException($"Arquivos EncTerrain não encontrados em {directory}");
        }

        if (mapId.HasValue)
        {
            foreach (var candidate in matches)
            {
                if (ExtractDigits(Path.GetFileName(candidate)) is { } digits && digits == mapId.Value)
                {
                    return candidate;
                }
            }
            throw new FileNotFoundException($"Nenhum arquivo EncTerrain correspondente ao mapa {mapId} foi encontrado em {directory}");
        }

        return matches[0];
    }

    private static string? GuessObjectFolder(string worldDirectory)
    {
        var info = new DirectoryInfo(worldDirectory);
        var name = info.Name;
        if (name.StartsWith("World", StringComparison.OrdinalIgnoreCase) && name.Length > 5)
        {
            var suffix = name[5..];
            var candidate = Path.Combine(info.Parent?.FullName ?? info.FullName, "Object" + suffix);
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
            var lowerCandidate = Path.Combine(info.Parent?.FullName ?? info.FullName, "object" + suffix);
            if (Directory.Exists(lowerCandidate))
            {
                return lowerCandidate;
            }
        }
        return null;
    }

    private static int? InferMapId(string worldDirectory)
    {
        for (var dir = new DirectoryInfo(worldDirectory); dir != null; dir = dir.Parent)
        {
            if (dir.Name.StartsWith("World", StringComparison.OrdinalIgnoreCase) && ExtractDigits(dir.Name) is { } digits)
            {
                return digits;
            }
        }
        return null;
    }

    private static int? ExtractDigits(string text)
    {
        var digits = new string(text.Where(char.IsDigit).ToArray());
        if (digits.Length == 0)
        {
            return null;
        }
        return int.Parse(digits, CultureInfo.InvariantCulture);
    }

    private static byte[] MapFileDecrypt(byte[] input)
    {
        var output = new byte[input.Length];
        byte wMapKey = 0x5E;
        for (var i = 0; i < input.Length; i++)
        {
            var value = input[i];
            var decrypted = (byte)(((value ^ XorKey[i % XorKey.Length]) - wMapKey) & 0xFF);
            output[i] = decrypted;
            wMapKey = (byte)((value + 0x3D) & 0xFF);
        }
        return output;
    }

    private static void BuxConvert(Span<byte> data)
    {
        for (var i = 0; i < data.Length; i++)
        {
            data[i] ^= BuxCode[i % BuxCode.Length];
        }
    }
}

internal sealed class TerrainData
{
    public TerrainData(float[] height, byte[] layer1, byte[] layer2, float[] alpha, ushort[] attributes)
    {
        Height = height;
        Layer1 = layer1;
        Layer2 = layer2;
        Alpha = alpha;
        Attributes = attributes;
    }

    public float[] Height { get; }
    public byte[] Layer1 { get; }
    public byte[] Layer2 { get; }
    public float[] Alpha { get; }
    public ushort[] Attributes { get; }
}

internal sealed class ObjectInstance
{
    public short TypeId { get; set; }
    public string? TypeName { get; set; }
    public Vector3 RawPosition { get; set; }
    public Vector3 Position { get; set; }
    public Vector3 Rotation { get; set; }
    public float Scale { get; set; }
}

internal sealed class WorldData
{
    public string WorldPath { get; init; } = string.Empty;
    public string ObjectsPath { get; init; } = string.Empty;
    public int MapId { get; init; }
    public int ObjectVersion { get; init; }
    public TerrainData Terrain { get; init; } = null!;
    public List<ObjectInstance> Objects { get; init; } = new();
}
