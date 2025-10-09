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
    private const float LoginSceneClassicHeightScale = 3.0f;
    private const float MinHeightBias = -500.0f;
    private const long ClassicHeightMinSize = 4 + 1080 + (long)TerrainSize * TerrainSize;
    private const long ExtendedHeightMinSize = 4 + 54 + (long)TerrainSize * TerrainSize * 3;
    private const int LoginSceneMapId = 55;
    private const int ObjectRecordSize = 2 + (3 * 4) + (3 * 4) + 4;

    private static readonly byte[] BuxCode = { 0xFC, 0xCF, 0xAB };

    public sealed class LoadOptions
    {
        public string? ObjectRoot { get; init; }
        public int? MapId { get; set; }
        public string? EnumPath { get; init; }
        public bool ForceExtendedHeight { get; init; }
        public float? HeightScale { get; init; }
        public bool LoadAttributes { get; init; } = true;
    }

    public WorldData Load(string worldDirectory, LoadOptions options)
    {
        if (!Directory.Exists(worldDirectory))
        {
            throw new DirectoryNotFoundException($"Diretório inválido: {worldDirectory}");
        }

        options.MapId ??= InferMapId(worldDirectory);

        string? attributesPath = null;
        if (options.LoadAttributes)
        {
            attributesPath = ResolveTerrainFile(worldDirectory, options.MapId, ".att");
        }
        var mappingPath = ResolveTerrainFile(worldDirectory, options.MapId, ".map");
        var (objectsPath, objectDirectory) = ResolveObjectResources(worldDirectory, options.ObjectRoot, options.MapId);
        var heightResource = ResolveHeightResource(worldDirectory, options.ForceExtendedHeight, options.MapId);
        var heightPath = heightResource.Path;
        var modelNames = LoadModelNames(options.EnumPath);

        var terrain = LoadTerrain(attributesPath, mappingPath, heightPath, heightResource.IsExtended, options.HeightScale, out var attributeMapId, out var mappingMapId);
        var objects = LoadObjects(objectsPath, modelNames, out var version, out var objectMapId);

        var resolvedMapId = options.MapId ?? (attributeMapId >= 0 ? attributeMapId : (mappingMapId >= 0 ? mappingMapId : objectMapId));
        var mapContext = MapContext.ForMapId(resolvedMapId);
        if (!heightResource.IsExtended && !options.HeightScale.HasValue && resolvedMapId == LoginSceneMapId)
        {
            terrain.ApplyHeightScale(LoginSceneClassicHeightScale);
        }
        AlignObjectsToTerrain(objects, terrain);

        var materialLibrary = new MaterialStateLibrary(worldDirectory, new[] { objectDirectory });
        var textureLibrary = new TextureLibrary(worldDirectory, objectDirectory, mapContext);
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

        var modelSearchRoots = new List<string>();
        void AddSearchRoot(string? candidate)
        {
            if (string.IsNullOrWhiteSpace(candidate))
            {
                return;
            }
            var full = Path.GetFullPath(candidate);
            if (Directory.Exists(full) && !modelSearchRoots.Any(path => string.Equals(path, full, StringComparison.OrdinalIgnoreCase)))
            {
                modelSearchRoots.Add(full);
            }
        }

        AddSearchRoot(objectDirectory);
        AddSearchRoot(Path.GetDirectoryName(objectsPath));
        AddSearchRoot(worldDirectory);
        var parent = Directory.GetParent(worldDirectory);
        if (parent != null)
        {
            AddSearchRoot(parent.FullName);
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
            ObjectDirectory = objectDirectory,
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
        const int headerSize = 4;
        if (decrypted.Length < headerSize)
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

        var payloadLength = span.Length - headerSize;
        var expectedPayload = (long)count * ObjectRecordSize;
        if (payloadLength < expectedPayload)
        {
            var expectedTotal = headerSize + expectedPayload;
            var expectedText = expectedTotal.ToString(CultureInfo.InvariantCulture);
            var actualText = span.Length.ToString(CultureInfo.InvariantCulture);
            var mapSuffix = mapId >= 0 ? mapId.ToString("D2", CultureInfo.InvariantCulture) : string.Empty;
            var objectLabel = string.IsNullOrEmpty(mapSuffix)
                ? "EncTerrainObjectXX.obj"
                : $"EncTerrainObject{mapSuffix}.obj";
            throw new InvalidDataException(
                $"Dados de objetos insuficientes: Esperado {expectedText} bytes no arquivo '{objectsPath}', mas havia {actualText}. " +
                $"Selecione o arquivo '{objectLabel}' correspondente à pasta selecionada.");
        }

        var objects = new List<ObjectInstance>(count);
        for (var i = 0; i < count; i++)
        {
            if (index + ObjectRecordSize > span.Length)
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

    private static TerrainData LoadTerrain(string? attributesPath, string mappingPath, string heightPath, bool isExtendedHeight, float? heightScale, out int attributeMapId, out int mappingMapId)
    {
        var height = new float[TerrainSize * TerrainSize];
        var layer1 = new byte[TerrainSize * TerrainSize];
        var layer2 = new byte[TerrainSize * TerrainSize];
        var alpha = new float[TerrainSize * TerrainSize];
        var attributes = new ushort[TerrainSize * TerrainSize];
        var hasAttributes = attributesPath is not null;
        float appliedScale;
        bool extended;

        // atributos
        if (attributesPath is not null)
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
        else
        {
            attributeMapId = -1;
            Array.Clear(attributes);
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
            extended = isExtendedHeight || Path.GetFileName(heightPath).Equals("TerrainHeightNew.OZB", StringComparison.OrdinalIgnoreCase);
            if (!extended)
            {
                var expected = 1080 + height.Length;
                if (payload.Length < expected)
                {
                    throw new InvalidDataException($"Arquivo de altura clássico truncado: {heightPath}");
                }
                var scale = heightScale ?? DefaultClassicHeightScale;
                appliedScale = scale;
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
                appliedScale = 1.0f;
                var pixel = payload.Slice(headerSize);
                for (var i = 0; i < height.Length; i++)
                {
                    var b = pixel[i * 3 + 0];
                    var g = pixel[i * 3 + 1];
                    var r = pixel[i * 3 + 2];
                    var combined = (r << 16) | (g << 8) | b;
                    var value = combined + MinHeightBias;
                    height[i] = value;
                }
            }
        }

        return new TerrainData(height, layer1, layer2, alpha, attributes, hasAttributes, extended, appliedScale);
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

    private static (string ObjectFile, string ObjectDirectory) ResolveObjectResources(string worldDirectory, string? objectRoot, int? mapId)
    {
        var normalizedRoot = NormalizeObjectRoot(objectRoot);
        var guessedRoot = GuessObjectFolder(worldDirectory);

        string? objectsFile = TryResolveObjectsInDirectory(worldDirectory, mapId);

        if (objectsFile is null && normalizedRoot is not null)
        {
            objectsFile = TryResolveObjectsInDirectory(normalizedRoot, mapId);
        }

        if (objectsFile is null && guessedRoot is not null && !PathsEqual(guessedRoot, normalizedRoot))
        {
            objectsFile = TryResolveObjectsInDirectory(guessedRoot, mapId);
        }

        if (objectsFile is null)
        {
            throw new FileNotFoundException("Arquivo EncTerrain*.obj não encontrado. Utilize --objects para informar a pasta correta.");
        }

        var objectDirectory = normalizedRoot
            ?? guessedRoot
            ?? Path.GetDirectoryName(objectsFile)
            ?? worldDirectory;

        return (Path.GetFullPath(objectsFile), Path.GetFullPath(objectDirectory));
    }

    private static string? TryResolveObjectsInDirectory(string directory, int? mapId)
    {
        try
        {
            return ResolveTerrainFile(directory, mapId, ".obj");
        }
        catch (DirectoryNotFoundException)
        {
            return null;
        }
        catch (FileNotFoundException)
        {
            return null;
        }
        catch (IOException)
        {
            return null;
        }

        return null;
    }

    private static string? NormalizeObjectRoot(string? objectRoot)
    {
        if (string.IsNullOrWhiteSpace(objectRoot))
        {
            return null;
        }

        var full = Path.GetFullPath(objectRoot);
        if (Directory.Exists(full))
        {
            return full;
        }

        if (File.Exists(full))
        {
            var directory = Path.GetDirectoryName(full);
            if (!string.IsNullOrEmpty(directory))
            {
                return Path.GetFullPath(directory);
            }
        }

        throw new DirectoryNotFoundException($"Diretório inválido: {objectRoot}");
    }

    private static bool PathsEqual(string? left, string? right)
    {
        if (left is null || right is null)
        {
            return false;
        }

        return string.Equals(Path.GetFullPath(left).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            Path.GetFullPath(right).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            StringComparison.OrdinalIgnoreCase);
    }

    private static HeightResource ResolveHeightResource(string worldDirectory, bool forceExtended, int? mapId)
    {
        var classicPath = Path.Combine(worldDirectory, "TerrainHeight.OZB");
        var extendedPath = Path.Combine(worldDirectory, "TerrainHeightNew.OZB");

        var classicSize = GetFileSize(classicPath);
        var extendedSize = GetFileSize(extendedPath);

        var classicValid = classicSize >= ClassicHeightMinSize;
        var extendedValid = extendedSize >= ExtendedHeightMinSize;

        var preferExtended = forceExtended || IsTerrainHeightExtendedMap(mapId);
        if (!preferExtended && ShouldPreferExtendedHeight(classicSize, extendedValid))
        {
            preferExtended = true;
        }

        if (preferExtended)
        {
            if (extendedValid)
            {
                return new HeightResource(extendedPath, true);
            }
            if (classicValid)
            {
                return new HeightResource(classicPath, false);
            }
        }
        else
        {
            if (classicValid)
            {
                return new HeightResource(classicPath, false);
            }
            if (extendedValid)
            {
                return new HeightResource(extendedPath, true);
            }
        }

        if (extendedValid)
        {
            return new HeightResource(extendedPath, true);
        }
        if (classicValid)
        {
            return new HeightResource(classicPath, false);
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

    private static bool IsTerrainHeightExtendedMap(int? mapId)
    {
        return mapId is 42 or 63 or 66;
    }

    private static bool ShouldPreferExtendedHeight(long classicSize, bool extendedValid)
    {
        return classicSize >= 0 && classicSize < ClassicHeightMinSize && extendedValid;
    }

    private static long GetFileSize(string path)
    {
        try
        {
            return new FileInfo(path).Length;
        }
        catch (IOException)
        {
            return -1;
        }
        catch (UnauthorizedAccessException)
        {
            return -1;
        }
    }

    private readonly struct HeightResource
    {
        public HeightResource(string path, bool isExtended)
        {
            Path = path;
            IsExtended = isExtended;
        }

        public string Path { get; }
        public bool IsExtended { get; }
    }
}

public sealed class TerrainData
{
    public TerrainData(float[] height, byte[] layer1, byte[] layer2, float[] alpha, ushort[] attributes, bool hasAttributes, bool usesExtendedHeight, float heightScale)
    {
        Height = height;
        Layer1 = layer1;
        Layer2 = layer2;
        Alpha = alpha;
        Attributes = attributes;
        HasAttributes = hasAttributes;
        UsesExtendedHeight = usesExtendedHeight;
        HeightScale = heightScale;
    }

    public float[] Height { get; }
    public byte[] Layer1 { get; }
    public byte[] Layer2 { get; }
    public float[] Alpha { get; }
    public ushort[] Attributes { get; }
    public bool HasAttributes { get; }
    public bool UsesExtendedHeight { get; }
    public float HeightScale { get; private set; }

    public void ApplyHeightScale(float newScale)
    {
        if (Math.Abs(newScale - HeightScale) < float.Epsilon)
        {
            return;
        }

        if (HeightScale <= 0f)
        {
            HeightScale = newScale;
            return;
        }

        var multiplier = newScale / HeightScale;
        for (var i = 0; i < Height.Length; i++)
        {
            Height[i] *= multiplier;
        }
        HeightScale = newScale;
    }
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
    public string ObjectDirectory { get; init; } = string.Empty;
    public int MapId { get; init; }
    public int ObjectVersion { get; init; }
    public TerrainData Terrain { get; init; } = null!;
    public List<ObjectInstance> Objects { get; init; } = new();
    public TerrainVisualData? Visual { get; init; }
    public ModelLibraryData ModelLibrary { get; init; } = new();
}
