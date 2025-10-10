using System.IO;

namespace MapWalker.Terrain;

internal sealed record MapDescriptor(int MapIndex, int WorldIndex, string DisplayName, DirectoryInfo WorldDirectory)
{
    public override string ToString() => $"{MapIndex:D2} - {DisplayName}";
}
