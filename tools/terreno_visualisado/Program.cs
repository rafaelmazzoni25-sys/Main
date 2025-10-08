using System.Globalization;
using TerrenoVisualisado.Core;

namespace TerrenoVisualisado;

internal static class Program
{
    private static int Main(string[] args)
    {
        try
        {
            var options = CliOptions.Parse(args);
            if (options.WorldDirectory is null)
            {
                throw new ArgumentException("Informe o diretório --world contendo os arquivos EncTerrain.");
            }

            var loader = new WorldLoader();
            var world = loader.Load(options.WorldDirectory, new WorldLoader.LoadOptions
            {
                ObjectRoot = options.ObjectDirectory,
                MapId = options.MapId,
                EnumPath = options.EnumPath,
                ForceExtendedHeight = options.ForceExtendedHeight,
                HeightScale = options.HeightScale,
            });

            PrintSummary(world);
            var output = options.OutputPath ?? "terrainsummary.json";
            WorldExporter.WriteJson(world, output);
            Console.WriteLine($"Resumo salvo em {output}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Erro: {ex.Message}");
            return 1;
        }
    }

    private static void PrintSummary(WorldData world)
    {
        Console.WriteLine($"Diretório: {world.WorldPath}");
        Console.WriteLine($"Mapa detectado: {world.MapId}");
        Console.WriteLine($"Objetos: {world.Objects.Count} (versão {world.ObjectVersion})");

        var tileCounts = new Dictionary<byte, int>();
        for (var i = 0; i < WorldLoader.TerrainSize * WorldLoader.TerrainSize; i++)
        {
            Increment(world.Terrain.Layer1[i]);
            Increment(world.Terrain.Layer2[i]);
        }

        Console.WriteLine("Top tiles por frequência:");
        foreach (var (tile, count) in tileCounts.OrderByDescending(kv => kv.Value).Take(8))
        {
            Console.WriteLine($"  Tile {tile:D3}: {count} ocorrências");
        }

        var attributeCounts = new Dictionary<ushort, int>();
        foreach (var attr in world.Terrain.Attributes)
        {
            attributeCounts.TryGetValue(attr, out var count);
            attributeCounts[attr] = count + 1;
        }
        Console.WriteLine("Atributos mais comuns:");
        foreach (var (attr, count) in attributeCounts.OrderByDescending(kv => kv.Value).Take(8))
        {
            Console.WriteLine($"  0x{attr:X3}: {count} tiles");
        }

        var objectCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var obj in world.Objects)
        {
            var key = obj.TypeName ?? $"ID_{obj.TypeId}";
            objectCounts.TryGetValue(key, out var count);
            objectCounts[key] = count + 1;
        }

        Console.WriteLine("Objetos estáticos mais frequentes:");
        foreach (var (name, count) in objectCounts.OrderByDescending(kv => kv.Value).Take(10))
        {
            Console.WriteLine($"  {name}: {count}");
        }

        void Increment(byte tile)
        {
            tileCounts.TryGetValue(tile, out var count);
            tileCounts[tile] = count + 1;
        }
    }

    private sealed record CliOptions
    {
        public string? WorldDirectory { get; init; }
        public string? ObjectDirectory { get; init; }
        public int? MapId { get; init; }
        public string? EnumPath { get; init; }
        public float? HeightScale { get; init; }
        public bool ForceExtendedHeight { get; init; }
        public string? OutputPath { get; init; }

        public static CliOptions Parse(string[] args)
        {
            var options = new CliOptions();
            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];
                switch (arg)
                {
                    case "--world":
                        options.WorldDirectory = RequireValue(args, ref i, arg);
                        break;
                    case "--objects":
                        options.ObjectDirectory = RequireValue(args, ref i, arg);
                        break;
                    case "--map":
                        options.MapId = int.Parse(RequireValue(args, ref i, arg), CultureInfo.InvariantCulture);
                        break;
                    case "--enum":
                        options.EnumPath = RequireValue(args, ref i, arg);
                        break;
                    case "--height-scale":
                        options.HeightScale = float.Parse(RequireValue(args, ref i, arg), CultureInfo.InvariantCulture);
                        break;
                    case "--extended-height":
                        options.ForceExtendedHeight = true;
                        break;
                    case "--output":
                        options.OutputPath = RequireValue(args, ref i, arg);
                        break;
                    default:
                        throw new ArgumentException($"Argumento desconhecido: {arg}");
                }
            }
            return options;
        }

        private static string RequireValue(IReadOnlyList<string> args, ref int index, string name)
        {
            if (index + 1 >= args.Count)
            {
                throw new ArgumentException($"O argumento {name} requer um valor.");
            }
            index++;
            return args[index];
        }
    }
}
