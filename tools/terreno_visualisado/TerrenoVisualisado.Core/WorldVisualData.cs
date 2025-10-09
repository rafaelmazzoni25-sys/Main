namespace TerrenoVisualisado.Core;

public sealed class TerrainVisualData
{
    public TextureImage? CompositeTexture { get; init; }
    public TextureImage? LitCompositeTexture { get; init; }
    public TextureImage? HighDetailCompositeTexture { get; init; }
    public TextureImage? HighDetailLitCompositeTexture { get; init; }
    public TextureImage? LightMap { get; init; }
    public string? LightMapPath { get; init; }
    public IReadOnlyDictionary<byte, string?> TileTextures { get; init; } = new Dictionary<byte, string?>();
    public IReadOnlyDictionary<byte, MaterialFlags> TileMaterialFlags { get; init; } = new Dictionary<byte, MaterialFlags>();
    public uint[] MaterialFlagsPerTile { get; init; } = Array.Empty<uint>();
    public IReadOnlyCollection<int> MissingTileIndices { get; init; } = Array.Empty<int>();
    public bool HasWaterTerrain { get; init; }
    public IReadOnlyDictionary<string, string?> SpecialTextures { get; init; } = new Dictionary<string, string?>();
}

public sealed class ModelLibraryData
{
    public IReadOnlyDictionary<short, BmdModel> Models { get; init; } = new Dictionary<short, BmdModel>();
    public IReadOnlyDictionary<short, string> Failures { get; init; } = new Dictionary<short, string>();
}
