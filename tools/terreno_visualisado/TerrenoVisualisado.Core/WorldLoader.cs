using System.Buffers.Binary;
using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;

namespace TerrenoVisualisado.Core;

public sealed class WorldLoader
{
    public const int TerrainSize = 256;
    public const float TerrainScale = 100.0f;
    private const float DefaultClassicHeightScale = 1.5f;
    private const float MinHeightBias = -500.0f;

    private static readonly byte[] BuxCode = { 0xFC, 0xCF, 0xAB };

    public sealed class LoadOptions
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
        var mapContext = MapContext.ForMapId(resolvedMapId);
        AlignObjectsToTerrain(objects, terrain);

        var materialLibrary = new MaterialStateLibrary(worldDirectory, new[] { objectsPath });
        var textureLibrary = new TextureLibrary(worldDirectory, objectsPath, mapContext);
        var tileIndices = new HashSet<byte>();
        foreach (var tile in terrain.Layer1)
        {
            tileIndices.Add(tile);
        }
        foreach (var tile in terrain.Layer2)
        {
            if (tile != 255)
            {
                tileIndices.Add(tile);
            }
        }

        var compositeTexture = textureLibrary.ComposeLayeredTexture(terrain.Layer1, terrain.Layer2, terrain.Alpha);
        var (lightMap, lightMapPath) = textureLibrary.LoadTerrainLight(mapContext);
        TextureImage? litComposite = null;
        if (lightMap is not null)
        {
            litComposite = ApplyLightMap(compositeTexture, lightMap);
        }
        var specialTextures = textureLibrary.LoadSpecialTextures(mapContext);
        var tileTextures = new Dictionary<byte, string?>(tileIndices.Count);
        var tileMaterials = new Dictionary<byte, MaterialFlags>(tileIndices.Count);
        foreach (var tile in tileIndices)
        {
            var resolvedTexture = textureLibrary.GetResolvedPath(tile) ?? textureLibrary.GetTileTextureName(tile);
            tileTextures[tile] = resolvedTexture;
            var materialState = materialLibrary.Lookup(resolvedTexture ?? string.Empty);
            tileMaterials[tile] = materialState.ToFlags();
        }
        var materialFlagsPerTile = ComputeTileMaterialFlags(terrain.Layer1, terrain.Layer2, terrain.Alpha, tileMaterials);

        var visual = new TerrainVisualData
        {
            CompositeTexture = compositeTexture,
            LitCompositeTexture = litComposite,
            LightMap = lightMap,
            LightMapPath = lightMapPath,
            TileTextures = tileTextures,
            TileMaterialFlags = tileMaterials,
            MaterialFlagsPerTile = materialFlagsPerTile,
            MissingTileIndices = textureLibrary.MissingIndices.ToArray(),
            HasWaterTerrain = mapContext.HasWaterTerrain,
            SpecialTextures = new Dictionary<string, string?>(specialTextures, StringComparer.OrdinalIgnoreCase),
        };

        var modelSearchRoots = new List<string>
        {
            objectsPath,
            worldDirectory,
        };
        var parent = Directory.GetParent(worldDirectory);
        if (parent != null)
        {
            modelSearchRoots.Add(parent.FullName);
        }
        var bmdLibrary = new BmdLibrary(modelSearchRoots, materialLibrary);
        var models = new Dictionary<short, BmdModel>();
        var modelFailures = new Dictionary<short, string>();
        foreach (var obj in objects)
        {
            if (models.ContainsKey(obj.TypeId) || modelFailures.ContainsKey(obj.TypeId))
            {
                continue;
            }
            var path = bmdLibrary.Resolve(obj);
            if (path is null)
            {
                modelFailures[obj.TypeId] = "Modelo não localizado";
                continue;
            }
            var model = bmdLibrary.Load(obj);
            if (model is not null)
            {
                models[obj.TypeId] = model;
            }
            else if (bmdLibrary.Failures.TryGetValue(path, out var failure))
            {
                modelFailures[obj.TypeId] = failure;
            }
        }

        var modelLibrary = new ModelLibraryData
        {
            Models = new Dictionary<short, BmdModel>(models),
            Failures = new Dictionary<short, string>(modelFailures),
        };

        return new WorldData
        {
            WorldPath = Path.GetFullPath(worldDirectory),
            ObjectsPath = Path.GetFullPath(objectsPath),
            MapId = resolvedMapId,
            ObjectVersion = version,
            Terrain = terrain,
            Objects = objects,
            Visual = visual,
            ModelLibrary = modelLibrary,
        };
    }

    private static uint[] ComputeTileMaterialFlags(byte[] layer1, byte[] layer2, float[] alpha, IReadOnlyDictionary<byte, MaterialFlags> tileMaterials)
    {
        var result = new uint[layer1.Length];
        for (var i = 0; i < layer1.Length; i++)
        {
            var baseTile = layer1[i];
            var overlayTile = layer2[i];
            var alphaValue = alpha[i];
            var baseFlags = tileMaterials.TryGetValue(baseTile, out var flags) ? flags : MaterialFlags.None;
            var overlayFlags = MaterialFlags.None;
            if (overlayTile != 255 && alphaValue > 0.01f && tileMaterials.TryGetValue(overlayTile, out var overlay))
            {
                overlayFlags = overlay;
            }
            result[i] = (uint)(baseFlags | overlayFlags);
        }
        return result;
    }

    private static TextureImage ApplyLightMap(TextureImage baseTexture, TextureImage lightMap)
    {
        var adjustedLight = lightMap;
        if (lightMap.Width != baseTexture.Width || lightMap.Height != baseTexture.Height)
        {
            adjustedLight = lightMap.Resize(baseTexture.Width, baseTexture.Height);
        }

        var resultPixels = new byte[baseTexture.Pixels.Length];
        for (var i = 0; i < baseTexture.Pixels.Length; i += 4)
        {
            var lr = adjustedLight.Pixels[i] / 255f;
            var lg = adjustedLight.Pixels[i + 1] / 255f;
            var lb = adjustedLight.Pixels[i + 2] / 255f;

            resultPixels[i] = (byte)Math.Clamp((int)Math.Round(baseTexture.Pixels[i] * lr), 0, 255);
            resultPixels[i + 1] = (byte)Math.Clamp((int)Math.Round(baseTexture.Pixels[i + 1] * lg), 0, 255);
            resultPixels[i + 2] = (byte)Math.Clamp((int)Math.Round(baseTexture.Pixels[i + 2] * lb), 0, 255);
            resultPixels[i + 3] = baseTexture.Pixels[i + 3];
        }

        return new TextureImage(baseTexture.Width, baseTexture.Height, resultPixels);
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

    public static float BilinearHeight(TerrainData terrain, float tileX, float tileY)
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
        var decrypted = MapCrypto.Decrypt(raw);
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
            var decrypted = MapCrypto.Decrypt(raw);
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
            var decrypted = MapCrypto.Decrypt(raw);
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
                    var combined = (r << 16) | (g << 8) | b;
                    var value = (combined / 10.0f) + MinHeightBias;
                    height[i] = value;
                }
            }
        }

        return new TerrainData(height, layer1, layer2, alpha, attributes);
    }

    private static Dictionary<int, string> LoadModelNames(string? enumPath)
    {
        var path = enumPath ?? "..\\data\\EnumModelType.eum";
        if (!File.Exists(path))
        {
            return new Dictionary<int, string>();
        }

        var regex = new Regex(@"^(?<id>\d+)\s*,\s*""(?<name>[^""]+)""", RegexOptions.Compiled);
        var map = new Dictionary<int, string>();
        foreach (var line in File.ReadLines(path))
        {
            var match = regex.Match(line);
            if (match.Success)
            {
                var id = int.Parse(match.Groups["id"].Value, CultureInfo.InvariantCulture);
                var name = match.Groups["name"].Value;
                map[id] = name;
            }
        }
        return map;
    }

    private static string ResolveObjectsPath(string worldDirectory, string? objectRoot, int? mapId)
    {
        if (!string.IsNullOrEmpty(objectRoot))
        {
            var explicitPath = ResolveTerrainFile(objectRoot, mapId, ".obj");
            if (!string.IsNullOrEmpty(explicitPath))
            {
                return explicitPath;
            }
        }

        if (GuessObjectFolder(worldDirectory) is { } guessed)
        {
            return ResolveTerrainFile(guessed, mapId, ".obj");
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

        var matches = EnumerateTerrainFiles(directory, extension);

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

    private static List<string> EnumerateTerrainFiles(string directory, string extension)
    {
        var basePattern = "EncTerrain*" + extension;
        var matches = EnumerateTerrainFiles(directory, basePattern, SearchOption.TopDirectoryOnly);
        if (matches.Count > 0)
        {
            return matches;
        }

        matches = EnumerateTerrainFiles(directory, basePattern, SearchOption.AllDirectories);
        if (matches.Count > 0)
        {
            return matches;
        }

        // Alguns pacotes personalizados renomeiam os arquivos de terreno com prefixos adicionais,
        // mantendo apenas "EncTerrain" no meio do nome. Para alinhar com o comportamento do
        // cliente original (que faz chamadas diretas ao nome base), use um curinga mais amplo
        // antes de desistir.
        matches = EnumerateTerrainFiles(directory, "*EncTerrain*" + extension, SearchOption.AllDirectories);

        return matches;
    }

    private static List<string> EnumerateTerrainFiles(string directory, string pattern, SearchOption option)
    {
        try
        {
            var enumerationOptions = new EnumerationOptions
            {
                IgnoreInaccessible = true,
                RecurseSubdirectories = option == SearchOption.AllDirectories,
#if NET6_0_OR_GREATER
                MatchCasing = MatchCasing.CaseInsensitive,
#endif
            };
            return Directory.EnumerateFiles(directory, pattern, enumerationOptions)
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToList();
        }
        catch (UnauthorizedAccessException)
        {
            return new List<string>();
        }
        catch (IOException)
        {
            return new List<string>();
        }
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

    private static void BuxConvert(Span<byte> data)
    {
        for (var i = 0; i < data.Length; i++)
        {
            data[i] ^= BuxCode[i % BuxCode.Length];
        }
    }
}

public sealed class TerrainData
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

public sealed class ObjectInstance
{
    public short TypeId { get; set; }
    public string? TypeName { get; set; }
    public Vector3 RawPosition { get; set; }
    public Vector3 Position { get; set; }
    public Vector3 Rotation { get; set; }
    public float Scale { get; set; }
}

public sealed class WorldData
{
    public string WorldPath { get; init; } = string.Empty;
    public string ObjectsPath { get; init; } = string.Empty;
    public int MapId { get; init; }
    public int ObjectVersion { get; init; }
    public TerrainData Terrain { get; init; } = null!;
    public List<ObjectInstance> Objects { get; init; } = new();
    public TerrainVisualData? Visual { get; init; }
    public ModelLibraryData ModelLibrary { get; init; } = new();
}
