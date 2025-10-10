namespace MapWalker.Terrain;

internal sealed class TerrainMesh
{
    public TerrainMesh(float[] positions, float[] normals, float[] texCoords, float[] tileCoords, uint[] indices)
    {
        Positions = positions;
        Normals = normals;
        TexCoords = texCoords;
        TileCoords = tileCoords;
        Indices = indices;
    }

    public float[] Positions { get; }
    public float[] Normals { get; }
    public float[] TexCoords { get; }
    public float[] TileCoords { get; }
    public uint[] Indices { get; }
}
