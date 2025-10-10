using OpenTK.Mathematics;

namespace MapWalker.Terrain;

internal sealed record ObjectInstance(short TypeId, Vector3 Position, Vector3 Rotation, float Scale);
