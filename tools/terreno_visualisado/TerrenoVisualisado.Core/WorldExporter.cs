using System.Text.Json;

namespace TerrenoVisualisado.Core;

public static class WorldExporter
{
    public static void WriteJson(WorldData world, string outputPath)
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
    }
}
