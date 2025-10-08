using System.Text.Json;

namespace TerrenoVisualisado.Core;

public static class WorldExporter
{
    public static void WriteJson(WorldData world, string outputPath)
    {
        var fullOutputPath = Path.GetFullPath(outputPath);
        var outputDirectory = Path.GetDirectoryName(fullOutputPath);
        if (!string.IsNullOrEmpty(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        string? textureAtlasRelative = null;
        if (world.Visual?.CompositeTexture is TextureImage atlas)
        {
            var baseName = Path.GetFileNameWithoutExtension(fullOutputPath) ?? "terrainsummary";
            var textureFileName = baseName + "_texture.png";
            var texturePath = string.IsNullOrEmpty(outputDirectory)
                ? textureFileName
                : Path.Combine(outputDirectory, textureFileName);
            atlas.SavePng(texturePath);
            textureAtlasRelative = textureFileName;
        }

        var visualPayload = world.Visual is null
            ? null
            : new
            {
                TextureAtlas = textureAtlasRelative,
                TileTextures = world.Visual.TileTextures.ToDictionary(kv => (int)kv.Key, kv => kv.Value),
                TileMaterialFlags = world.Visual.TileMaterialFlags.ToDictionary(
                    kv => (int)kv.Key,
                    kv => new
                    {
                        Numeric = (uint)kv.Value,
                        Flags = kv.Value.ToString(),
                    }),
                world.Visual.MaterialFlagsPerTile,
                MissingTileIndices = world.Visual.MissingTileIndices,
            };

        var modelsPayload = world.ModelLibrary.Models.ToDictionary(
            kv => (int)kv.Key,
            kv => new
            {
                kv.Value.Name,
                kv.Value.Version,
                kv.Value.SourcePath,
                Meshes = kv.Value.Meshes.Select(mesh => new
                {
                    mesh.Name,
                    mesh.TextureName,
                    MaterialFlags = new
                    {
                        Numeric = (uint)mesh.MaterialFlags,
                        Flags = mesh.MaterialFlags.ToString(),
                    },
                    mesh.Positions,
                    mesh.Normals,
                    mesh.TexCoords,
                    mesh.Indices,
                    BoneIndices = mesh.BoneIndices.Select(b => (int)b),
                }),
                Actions = kv.Value.Actions.Select(action => new
                {
                    action.KeyframeCount,
                    action.LockPositions,
                    LockedPositions = action.LockPositions
                        ? action.LockedPositions.Select(p => new[] { p.X, p.Y, p.Z })
                        : Array.Empty<float[]>(),
                }),
                Bones = kv.Value.Bones.Select(bone => new
                {
                    bone.Name,
                    bone.Parent,
                    bone.IsDummy,
                    Animations = bone.Animations.Select(animation => new
                    {
                        Positions = animation.Positions.Select(p => new[] { p.X, p.Y, p.Z }),
                        Rotations = animation.Rotations.Select(r => new[] { r.X, r.Y, r.Z }),
                        Quaternions = animation.Quaternions.Select(q => new[] { q.X, q.Y, q.Z, q.W }),
                    }),
                }),
            });

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
            Visual = visualPayload,
            Models = new
            {
                Loaded = modelsPayload,
                Failures = world.ModelLibrary.Failures.ToDictionary(kv => (int)kv.Key, kv => kv.Value),
            },
        };

        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
        };
        File.WriteAllText(fullOutputPath, JsonSerializer.Serialize(export, options));
    }
}
