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
            if (options.ShowHelp)
            {
                PrintUsage();
                return 0;
            }
            if (options.WorldDirectory is null)
            {
                PrintUsage();
                Console.Error.WriteLine("Erro: Informe o diretório --world contendo os arquivos EncTerrain.");
                return 1;
            }

            var loader = new WorldLoader();
            var world = loader.Load(options.WorldDirectory, new WorldLoader.LoadOptions
            {
                ObjectRoot = options.ObjectDirectory,
                MapId = options.MapId,
                EnumPath = options.EnumPath,
                ForceExtendedHeight = options.ForceExtendedHeight,
                HeightScale = options.HeightScale,
                LoadAttributes = !options.SkipAttributes,
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
        Console.WriteLine($"EncTerrain: {world.ObjectsPath}");
        Console.WriteLine($"Object dir: {world.ObjectDirectory}");
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

        if (world.Terrain.HasAttributes)
        {
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
        }
        else
        {
            Console.WriteLine("Atributos: não carregados");
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

        if (world.Visual is { } visual)
        {
            Console.WriteLine($"Texturas distintas: {visual.TileTextures.Count} (faltando {visual.MissingTileIndices.Count})");
            var waterTiles = visual.MaterialFlagsPerTile.Count(flag => (flag & (uint)MaterialFlags.Water) != 0);
            var lavaTiles = visual.MaterialFlagsPerTile.Count(flag => (flag & (uint)MaterialFlags.Lava) != 0);
            Console.WriteLine($"Tiles com água: {waterTiles}, lava: {lavaTiles}");
            if (visual.LightMap is not null)
            {
                Console.WriteLine($"TerrainLight aplicado: {visual.LightMapPath ?? "arquivo detectado"}");
            }
            else
            {
                Console.WriteLine("TerrainLight aplicado: nenhum arquivo encontrado");
            }
            Console.WriteLine($"Mapa com CreateWaterTerrain: {(visual.HasWaterTerrain ? "sim" : "não")}");
            if (visual.SpecialTextures.Count > 0)
            {
                var loaded = visual.SpecialTextures.Count(kv => !string.IsNullOrEmpty(kv.Value));
                var missing = visual.SpecialTextures.Count - loaded;
                Console.WriteLine($"Texturas especiais carregadas: {loaded}, ausentes: {missing}");
            }
        }

        Console.WriteLine($"Modelos BMD carregados: {world.ModelLibrary.Models.Count} (falhas: {world.ModelLibrary.Failures.Count})");
        if (world.ModelLibrary.Models.Count > 0)
        {
            var animatedModels = world.ModelLibrary.Models.Values.Count(model => model.Actions.Any(action => action.KeyframeCount > 0));
            var totalBones = world.ModelLibrary.Models.Values.Sum(model => model.Bones.Count(bone => !bone.IsDummy));
            var totalKeyframes = world.ModelLibrary.Models.Values
                .SelectMany(model => model.Actions)
                .Sum(action => action.KeyframeCount);
            Console.WriteLine($"Modelos com animação: {animatedModels}, ossos ativos: {totalBones}, quadros: {totalKeyframes}");
        }

        void Increment(byte tile)
        {
            tileCounts.TryGetValue(tile, out var count);
            tileCounts[tile] = count + 1;
        }
    }

    private static void PrintUsage()
    {
        Console.WriteLine("TerrenoVisualisado - utilitário de linha de comando");
        Console.WriteLine();
        Console.WriteLine("Uso:");
        Console.WriteLine("  TerrenoVisualisado --world <pasta> [opções]");
        Console.WriteLine();
        Console.WriteLine("Opções disponíveis:");
        Console.WriteLine("  --world <pasta>         Diretório contendo os arquivos EncTerrain");
        Console.WriteLine("  --objects <pasta>       Diretório ObjectX correspondente (opcional)");
        Console.WriteLine("  --map <id>              Força o ID numérico do mapa");
        Console.WriteLine("  --enum <arquivo>        Caminho para o arquivo _enum.h");
        Console.WriteLine("  --height-scale <valor>  Fator aplicado ao TerrainHeight.OZB");
        Console.WriteLine("  --extended-height       Usa TerrainHeightNew.OZB mesmo se o clássico existir");
        Console.WriteLine("  --skip-attributes       Não lê EncTerrain*.att");
        Console.WriteLine("  --output <arquivo>      Nome do arquivo JSON de saída (padrão terrainsummary.json)");
        Console.WriteLine("  --help                  Mostra este resumo e sai");
        Console.WriteLine();
        Console.WriteLine("Exemplo:");
        Console.WriteLine("  TerrenoVisualisado --world C:/MuOnline/Data/World1 --objects C:/MuOnline/Data/Object1");
    }

    private sealed record CliOptions
    {
        public string? WorldDirectory { get; set; }
        public string? ObjectDirectory { get; set; }
        public int? MapId { get; set; }
        public string? EnumPath { get; set; }
        public float? HeightScale { get; set; }
        public bool ForceExtendedHeight { get; set; }
        public string? OutputPath { get; set; }
        public bool ShowHelp { get; set; }
        public bool SkipAttributes { get; set; }

        public static CliOptions Parse(string[] args)
        {
            var options = new CliOptions();
            if (args.Length == 0)
            {
                options.ShowHelp = true;
                return options;
            }
            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];
                switch (arg)
                {
                    case "--help":
                    case "-h":
                        options.ShowHelp = true;
                        break;
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
                    case "--skip-attributes":
                        options.SkipAttributes = true;
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
