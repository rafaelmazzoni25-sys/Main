namespace TerrenoVisualisado.Core;

[System.Flags]
public enum MaterialFlags : uint
{
    None = 0,
    Water = 1u << 0,
    Lava = 1u << 1,
    Transparent = 1u << 2,
    Additive = 1u << 3,
    Emissive = 1u << 4,
    AlphaTest = 1u << 5,
    DoubleSided = 1u << 6,
    NoShadow = 1u << 7,
    NormalMap = 1u << 8,
}
