using System;
using System.Globalization;

namespace TerrenoVisualisado.Core;

public sealed class TextureLibrary
{
    private static readonly Dictionary<int, string[]> BaseTileTextureCandidates = new()
    {
        [0] = new[] { "TileGrass01", "TileGrass01_R" },
        [1] = new[] { "TileGrass02" },
        [2] = new[] { "TileGround01", "AlphaTileGround01", "AlphaTile01" },
        [3] = new[] { "TileGround02", "AlphaTileGround02" },
        [4] = new[] { "TileGround03", "AlphaTileGround03" },
        [5] = new[] { "TileWater01", "Object25/water1", "Object25/water2" },
        [6] = new[] { "TileWood01" },
        [7] = new[] { "TileRock01" },
        [8] = new[] { "TileRock02" },
        [9] = new[] { "TileRock03" },
        [10] = new[] { "TileRock04", "AlphaTile01" },
        [11] = new[] { "TileRock05", "Object64/song_lava1" },
        [12] = new[] { "TileRock06", "AlphaTile01" },
        [13] = new[] { "TileRock07" },
    };

    static TextureLibrary()
    {
        for (var ext = 1; ext <= 16; ext++)
        {
            BaseTileTextureCandidates[13 + ext] = new[] { $"ExtTile{ext:00}" };
        }
    }

    private readonly List<string> _searchRoots;
    private readonly Dictionary<int, TextureImage?> _cache = new();
    private readonly Dictionary<(int Index, int Size), TextureImage> _resized = new();
    private readonly HashSet<int> _missing = new();
    private readonly Dictionary<int, string> _resolvedPaths = new();
    private readonly int _detailFactor;
    private readonly MapContext _context;
    private readonly string _worldFolderName;
    private readonly Dictionary<string, (TextureImage? Image, string? Path)> _arbitraryCache = new(StringComparer.OrdinalIgnoreCase);

    public TextureLibrary(string worldDirectory, string? objectDirectory, MapContext context, int detailFactor = 2)
    {
        _detailFactor = Math.Max(1, detailFactor);
        _context = context;
        _worldFolderName = Path.GetFileName(Path.GetFullPath(worldDirectory).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        _searchRoots = BuildSearchRoots(worldDirectory, objectDirectory, context);
    }

    public int DetailFactor => _detailFactor;

    public IReadOnlyCollection<int> MissingIndices => _missing;

    public string? GetResolvedPath(int index)
    {
        return _resolvedPaths.TryGetValue(index, out var path) ? path : null;
    }

    public string? GetTileTextureName(int index)
    {
        var candidates = EnumerateTileCandidates(index).ToArray();
        if (candidates.Length > 0)
        {
            return candidates[0];
        }
        return $"ExtTile{index:00}";
    }

    public TextureImage ComposeLayeredTexture(byte[] layer1, byte[] layer2, float[] alpha, int? detailFactorOverride = null)
    {
        var size = WorldLoader.TerrainSize;
        if (layer1.Length != size * size || layer2.Length != size * size || alpha.Length != size * size)
        {
            throw new ArgumentException("Camadas inválidas");
        }

        var patch = detailFactorOverride.HasValue ? Math.Max(1, detailFactorOverride.Value) : _detailFactor;
        var width = size * patch;
        var pixels = new byte[width * width * 4];

        for (var tileY = 0; tileY < size; tileY++)
        {
            for (var tileX = 0; tileX < size; tileX++)
            {
                var tileIndex = tileY * size + tileX;
                var baseIndex = layer1[tileIndex];
                var overlayIndex = layer2[tileIndex];
                var alphaValue = alpha[tileIndex];
                var basePatch = ResolvePatch(baseIndex, patch);
                var overlayPatch = overlayIndex != 255 ? ResolvePatch(overlayIndex, patch) : null;
                var mode = overlayPatch is not null && alphaValue >= 0.999f ? 2 : (overlayPatch is not null && alphaValue > 0f ? 1 : 0);
                BlendTile(pixels, basePatch, overlayPatch, alphaValue, mode, width, patch, tileX, tileY);
            }
        }

        return new TextureImage(width, width, pixels);
    }

    public int EstimateDetailFactor(int minimum = 8, int maximum = 32)
    {
        minimum = Math.Max(1, minimum);
        maximum = Math.Max(minimum, maximum);

        var largest = 0;
        foreach (var entry in _cache.Values)
        {
            if (entry is null)
            {
                continue;
            }

            var dimension = Math.Max(entry.Width, entry.Height);
            if (dimension > largest)
            {
                largest = dimension;
            }
        }

        if (largest <= 0)
        {
            return minimum;
        }

        var recommended = 1;
        while (recommended < largest)
        {
            recommended <<= 1;
        }

        if (recommended < minimum)
        {
            recommended = minimum;
        }
        else if (recommended > maximum)
        {
            recommended = maximum;
        }

        return recommended;
    }

    private void BlendTile(byte[] canvas, TextureImage basePatch, TextureImage? overlay, float alpha, int mode, int canvasWidth, int patchSize, int tileX, int tileY)
    {
        var startX = tileX * patchSize;
        var startY = tileY * patchSize;
        for (var y = 0; y < patchSize; y++)
        {
            for (var x = 0; x < patchSize; x++)
            {
                var canvasOffset = ((startY + y) * canvasWidth + (startX + x)) * 4;
                var baseOffset = (y * patchSize + x) * 4;
                var r = basePatch.Pixels[baseOffset + 0];
                var g = basePatch.Pixels[baseOffset + 1];
                var b = basePatch.Pixels[baseOffset + 2];
                var a = basePatch.Pixels[baseOffset + 3];
                if (overlay is not null && mode != 0)
                {
                    var overlayOffset = baseOffset;
                    var or = overlay.Pixels[overlayOffset + 0];
                    var og = overlay.Pixels[overlayOffset + 1];
                    var ob = overlay.Pixels[overlayOffset + 2];
                    var oa = overlay.Pixels[overlayOffset + 3];
                    if (mode == 2)
                    {
                        r = or; g = og; b = ob; a = oa;
                    }
                    else
                    {
                        var blend = Math.Clamp(alpha, 0f, 1f);
                        r = (byte)Math.Clamp((int)Math.Round(r * (1 - blend) + or * blend), 0, 255);
                        g = (byte)Math.Clamp((int)Math.Round(g * (1 - blend) + og * blend), 0, 255);
                        b = (byte)Math.Clamp((int)Math.Round(b * (1 - blend) + ob * blend), 0, 255);
                        a = (byte)Math.Clamp((int)Math.Round(a * (1 - blend) + oa * blend), 0, 255);
                    }
                }
                canvas[canvasOffset + 0] = r;
                canvas[canvasOffset + 1] = g;
                canvas[canvasOffset + 2] = b;
                canvas[canvasOffset + 3] = a;
            }
        }
    }

    private TextureImage ResolvePatch(int index, int patchSize)
    {
        var key = (index, patchSize);
        if (_resized.TryGetValue(key, out var cached))
        {
            return cached;
        }

        var image = LoadTexture(index);
        TextureImage patch;
        if (image is null)
        {
            var fallback = BuildFallbackColor(index);
            patch = TextureImage.FromRgba(patchSize, patchSize, fallback);
        }
        else
        {
            patch = image.Resize(patchSize, patchSize);
        }
        _resized[key] = patch;
        return patch;
    }

    private TextureImage? LoadTexture(int index)
    {
        if (_cache.TryGetValue(index, out var cached))
        {
            return cached;
        }

        var candidates = EnumerateTileCandidates(index);

        foreach (var candidate in candidates)
        {
            foreach (var root in _searchRoots)
            {
                foreach (var path in TextureFileLoader.EnumerateCandidatePaths(root, candidate))
                {
                    if (TextureFileLoader.TryLoad(path, out var image))
                    {
                        _cache[index] = image;
                        _resolvedPaths[index] = path;
                        return image;
                    }
                }
            }
        }

        _missing.Add(index);
        _cache[index] = null;
        return null;
    }

    public (TextureImage? Image, string? Path) LoadArbitrary(string baseName)
    {
        if (_arbitraryCache.TryGetValue(baseName, out var cached))
        {
            return cached;
        }

        foreach (var root in _searchRoots)
        {
            foreach (var path in TextureFileLoader.EnumerateCandidatePaths(root, baseName))
            {
                if (TextureFileLoader.TryLoad(path, out var image))
                {
                    var result = (image, path);
                    _arbitraryCache[baseName] = result;
                    return result;
                }
            }
        }

        var fallback = ((TextureImage?)null, (string?)null);
        _arbitraryCache[baseName] = fallback;
        return fallback;
    }

    public (TextureImage? Image, string? Path) LoadTerrainLight(MapContext context)
    {
        foreach (var candidate in context.EnumerateTerrainLightCandidates())
        {
            var (image, path) = LoadArbitrary(Path.Combine(_worldFolderName, candidate));
            if (image is not null)
            {
                return (image, path);
            }
        }

        foreach (var candidate in context.EnumerateTerrainLightCandidates())
        {
            var (image, path) = LoadArbitrary(candidate);
            if (image is not null)
            {
                return (image, path);
            }
        }

        return (null, null);
    }

    public IReadOnlyDictionary<string, string?> LoadSpecialTextures(MapContext context)
    {
        var result = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);

        string? LoadFirst(params string[] candidates)
        {
            foreach (var candidate in candidates)
            {
                if (string.IsNullOrWhiteSpace(candidate))
                {
                    continue;
                }
                var (image, path) = LoadArbitrary(candidate);
                if (image is not null || path is not null)
                {
                    return path;
                }
            }
            return null;
        }

        if (!string.IsNullOrEmpty(_worldFolderName))
        {
            result["Leaf01"] = LoadFirst(Path.Combine(_worldFolderName, "leaf01"));
            result["Leaf02"] = LoadFirst(Path.Combine(_worldFolderName, "leaf02"));
        }
        else
        {
            result["Leaf01"] = LoadFirst("leaf01");
            result["Leaf02"] = LoadFirst("leaf02");
        }

        if (context.IsCryWolf)
        {
            result["Rain01"] = LoadFirst(Path.Combine("World1", "rain011"), Path.Combine("World1", "rain01"));
        }
        else
        {
            result["Rain01"] = LoadFirst(Path.Combine("World1", "rain01"));
        }

        result["Rain02"] = LoadFirst(Path.Combine("World1", "rain02"));
        result["Rain03"] = LoadFirst(Path.Combine("World10", "rain03"));

        return result;
    }

    private static byte[] BuildFallbackColor(int index)
    {
        var random = (index * 2654435761u) ^ 0xA53B5E2Du;
        var r = (byte)(random & 0xFF);
        var g = (byte)((random >> 8) & 0xFF);
        var b = (byte)((random >> 16) & 0xFF);
        var a = (byte)255;
        var size = 4;
        var buffer = new byte[size * size * 4];
        for (var i = 0; i < size * size; i++)
        {
            var offset = i * 4;
            buffer[offset + 0] = r;
            buffer[offset + 1] = g;
            buffer[offset + 2] = b;
            buffer[offset + 3] = a;
        }
        return buffer;
    }

    private static List<string> BuildSearchRoots(string worldDirectory, string? objectDirectory, MapContext context)
    {
        var roots = new List<string>();
        void Add(string? path)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                return;
            }
            var full = Path.GetFullPath(path);
            if (Directory.Exists(full) && !roots.Contains(full, StringComparer.OrdinalIgnoreCase))
            {
                roots.Add(full);
            }
        }

        Add(worldDirectory);
        var worldParent = Directory.GetParent(worldDirectory);
        if (worldParent != null)
        {
            Add(worldParent.FullName);
            foreach (var child in worldParent.EnumerateDirectories())
            {
                var name = child.Name.ToLowerInvariant();
                if (name.StartsWith("world") || name.StartsWith("object") || name.StartsWith("texture"))
                {
                    Add(child.FullName);
                }
            }
        }
        Add(objectDirectory);

        if (context.HasValidMapId && worldParent != null)
        {
            var digits = context.MapId.ToString(CultureInfo.InvariantCulture);
            foreach (var child in worldParent.EnumerateDirectories())
            {
                if (child.Name.Contains(digits, StringComparison.OrdinalIgnoreCase))
                {
                    Add(child.FullName);
                }
            }
        }

        return roots;
    }

    private IEnumerable<string> EnumerateTileCandidates(int index)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var list = new List<string>();

        void AddCandidate(string candidate)
        {
            if (string.IsNullOrWhiteSpace(candidate))
            {
                return;
            }
            if (seen.Add(candidate))
            {
                list.Add(candidate);
            }
        }

        switch (index)
        {
            case 0:
                if (_context.IsPkField || _context.IsDoppelGanger2)
                {
                    AddCandidate("TileGrass01_R");
                }
                AddCandidate("TileGrass01");
                AddCandidate("TileGrass01_R");
                break;
            case 2:
                if (_context.UsesAlphaGround01)
                {
                    AddCandidate("AlphaTileGround01");
                }
                AddCandidate("TileGround01");
                AddCandidate("AlphaTileGround01");
                AddCandidate("AlphaTile01");
                break;
            case 3:
                if (_context.IsKanturuThird)
                {
                    AddCandidate("AlphaTileGround02");
                }
                AddCandidate("TileGround02");
                AddCandidate("AlphaTileGround02");
                break;
            case 4:
                if (_context.IsCursedTemple)
                {
                    AddCandidate("AlphaTileGround03");
                }
                AddCandidate("TileGround03");
                AddCandidate("AlphaTileGround03");
                break;
            case 10:
                if (_context.UsesAlphaTileForRock04)
                {
                    AddCandidate("AlphaTile01");
                }
                AddCandidate("TileRock04");
                AddCandidate("AlphaTile01");
                break;
            case 11:
                if (_context.IsPkField || _context.IsDoppelGanger2)
                {
                    AddCandidate("Object64/song_lava1");
                }
                AddCandidate("TileRock05");
                break;
            case 12:
                if (_context.IsKarutan)
                {
                    AddCandidate("AlphaTile01");
                }
                AddCandidate("TileRock06");
                AddCandidate("AlphaTile01");
                break;
        }

        if (BaseTileTextureCandidates.TryGetValue(index, out var baseCandidates))
        {
            foreach (var candidate in baseCandidates)
            {
                AddCandidate(candidate);
            }
        }

        if (list.Count == 0)
        {
            AddCandidate($"ExtTile{index:00}");
        }

        return list;
    }
}
