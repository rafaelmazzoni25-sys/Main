using System.Text.Json;
using System.Text.Json.Nodes;

namespace TerrenoVisualisado.Core;

public sealed record MaterialState
{
    public string Name { get; init; } = string.Empty;
    public string SrcBlend { get; init; } = "one";
    public string DstBlend { get; init; } = "zero";
    public string BlendOp { get; init; } = "add";
    public bool AlphaTest { get; init; }
    public string AlphaFunc { get; init; } = "always";
    public float AlphaRef { get; init; }
    public bool DepthWrite { get; init; } = true;
    public bool DepthTest { get; init; } = true;
    public string CullMode { get; init; } = "back";
    public (float R, float G, float B) Emissive { get; init; } = (0f, 0f, 0f);
    public (float R, float G, float B) Specular { get; init; } = (0.2f, 0.2f, 0.2f);
    public float SpecularPower { get; init; } = 16f;
    public bool DoubleSided { get; init; }
    public bool Additive { get; init; }
    public bool Transparent { get; init; }
    public bool Water { get; init; }
    public bool Lava { get; init; }
    public string? NormalMap { get; init; }
    public bool ReceiveShadows { get; init; } = true;

    public MaterialFlags ToFlags()
    {
        var flags = MaterialFlags.None;
        if (Water)
        {
            flags |= MaterialFlags.Water | MaterialFlags.Transparent;
        }
        if (Lava)
        {
            flags |= MaterialFlags.Lava | MaterialFlags.Transparent;
        }
        if (Transparent)
        {
            flags |= MaterialFlags.Transparent;
        }
        if (Additive)
        {
            flags |= MaterialFlags.Additive;
        }
        if (MathF.Sqrt(Emissive.R * Emissive.R + Emissive.G * Emissive.G + Emissive.B * Emissive.B) > 1e-3f)
        {
            flags |= MaterialFlags.Emissive;
        }
        if (AlphaTest)
        {
            flags |= MaterialFlags.AlphaTest;
        }
        if (DoubleSided || string.Equals(CullMode, "none", StringComparison.OrdinalIgnoreCase))
        {
            flags |= MaterialFlags.DoubleSided;
        }
        if (!ReceiveShadows)
        {
            flags |= MaterialFlags.NoShadow;
        }
        if (!string.IsNullOrEmpty(NormalMap))
        {
            flags |= MaterialFlags.NormalMap;
        }
        return flags;
    }
}

public sealed class MaterialStateLibrary
{
    private static readonly string[] CandidateFiles =
    {
        "materials.json",
        "material_table.json",
        "material_table.txt",
        "material_table.csv",
        "materialstate.json",
        "materialstate.txt",
    };

    private readonly Dictionary<string, MaterialState> _states = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _aliases = new(StringComparer.OrdinalIgnoreCase);

    public MaterialStateLibrary(string? worldDirectory, IEnumerable<string>? extraRoots = null)
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
        if (worldDirectory != null)
        {
            var parent = Directory.GetParent(worldDirectory);
            if (parent != null)
            {
                Add(Path.Combine(parent.FullName, "Texture"));
            }
        }
        if (extraRoots != null)
        {
            foreach (var root in extraRoots)
            {
                Add(root);
            }
        }

        foreach (var root in roots)
        {
            foreach (var candidate in CandidateFiles)
            {
                var file = Path.Combine(root, candidate);
                if (!File.Exists(file))
                {
                    continue;
                }
                try
                {
                    LoadTable(file);
                }
                catch
                {
                    // ignore malformed tables
                }
            }
        }
    }

    public MaterialState Lookup(string textureName)
    {
        if (string.IsNullOrWhiteSpace(textureName))
        {
            return BuildFallback(textureName);
        }

        var normalized = NormalizeKey(textureName);
        if (_states.TryGetValue(normalized, out var state))
        {
            return state;
        }

        var stem = Path.GetFileNameWithoutExtension(normalized);
        if (!string.IsNullOrEmpty(stem) && _states.TryGetValue(stem, out state))
        {
            return state;
        }

        if (_aliases.TryGetValue(normalized, out var alias) && _states.TryGetValue(alias, out state))
        {
            return state;
        }

        return BuildFallback(textureName);
    }

    private void LoadTable(string path)
    {
        var extension = Path.GetExtension(path);
        if (string.Equals(extension, ".json", StringComparison.OrdinalIgnoreCase))
        {
            LoadJson(path);
        }
        else if (string.Equals(extension, ".csv", StringComparison.OrdinalIgnoreCase))
        {
            LoadCsv(path);
        }
        else
        {
            LoadText(path);
        }
    }

    private void LoadJson(string path)
    {
        using var stream = File.OpenRead(path);
        var node = JsonNode.Parse(stream);
        if (node is JsonObject obj)
        {
            foreach (var (key, value) in obj)
            {
                if (value is JsonObject nested)
                {
                    StoreEntry(key, nested);
                }
            }
        }
        else if (node is JsonArray array)
        {
            foreach (var element in array)
            {
                if (element is JsonObject nested && nested.TryGetPropertyValue("name", out var nameNode) && nameNode is JsonValue nameValue && nameValue.TryGetValue<string>(out var name))
                {
                    StoreEntry(name, nested);
                }
            }
        }
    }

    private void LoadCsv(string path)
    {
        using var reader = new StreamReader(path);
        var header = reader.ReadLine();
        if (header is null)
        {
            return;
        }
        var columns = header.Split(',');
        int IndexOf(string column)
        {
            for (var i = 0; i < columns.Length; i++)
            {
                if (string.Equals(columns[i].Trim(), column, StringComparison.OrdinalIgnoreCase))
                {
                    return i;
                }
            }
            return -1;
        }

        var nameIndex = IndexOf("name");
        if (nameIndex < 0)
        {
            nameIndex = IndexOf("texture");
        }
        if (nameIndex < 0)
        {
            return;
        }

        string? ReadValue(string[] parts, string column)
        {
            var idx = IndexOf(column);
            if (idx >= 0 && idx < parts.Length)
            {
                return parts[idx].Trim();
            }
            return null;
        }

        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }
            var parts = line.Split(',');
            if (nameIndex >= parts.Length)
            {
                continue;
            }
            var name = parts[nameIndex].Trim();
            if (string.IsNullOrEmpty(name))
            {
                continue;
            }

            var payload = new Dictionary<string, object?>
            {
                ["src_blend"] = ReadValue(parts, "src_blend"),
                ["dst_blend"] = ReadValue(parts, "dst_blend"),
                ["alpha_test"] = ReadValue(parts, "alpha_test"),
                ["alpha_ref"] = ReadValue(parts, "alpha_ref"),
                ["depth_write"] = ReadValue(parts, "depth_write"),
                ["depth_test"] = ReadValue(parts, "depth_test"),
                ["cull_mode"] = ReadValue(parts, "cull_mode"),
                ["additive"] = ReadValue(parts, "additive"),
                ["transparent"] = ReadValue(parts, "transparent"),
                ["water"] = ReadValue(parts, "water"),
                ["lava"] = ReadValue(parts, "lava"),
            };
            StoreEntry(name, payload);
        }
    }

    private void LoadText(string path)
    {
        foreach (var rawLine in File.ReadLines(path))
        {
            var line = rawLine.Trim();
            if (string.IsNullOrEmpty(line) || line.StartsWith('#'))
            {
                continue;
            }

            if (line.Contains(':'))
            {
                var parts = line.Split(':', 2);
                if (parts.Length == 2)
                {
                    StoreEntry(parts[0].Trim(), new Dictionary<string, object?> { ["params"] = parts[1].Trim() });
                }
                continue;
            }

            var tokens = line.Split(',');
            if (tokens.Length == 0)
            {
                continue;
            }

            var name = tokens[0].Trim();
            if (string.IsNullOrEmpty(name))
            {
                continue;
            }

            var payload = new Dictionary<string, object?>
            {
                ["src_blend"] = tokens.Length > 1 ? tokens[1].Trim() : null,
                ["dst_blend"] = tokens.Length > 2 ? tokens[2].Trim() : null,
                ["alpha_test"] = tokens.Length > 3 ? tokens[3].Trim() : null,
                ["alpha_ref"] = tokens.Length > 4 ? tokens[4].Trim() : null,
                ["depth_write"] = tokens.Length > 5 ? tokens[5].Trim() : null,
                ["depth_test"] = tokens.Length > 6 ? tokens[6].Trim() : null,
                ["cull_mode"] = tokens.Length > 7 ? tokens[7].Trim() : null,
            };
            StoreEntry(name, payload);
        }
    }

    private void StoreEntry(string name, IReadOnlyDictionary<string, object?> payload)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            return;
        }

        var state = new MaterialState
        {
            Name = NormalizeKey(name),
            SrcBlend = payload.TryGetValue("src_blend", out var src) ? SafeString(src, "one") : "one",
            DstBlend = payload.TryGetValue("dst_blend", out var dst) ? SafeString(dst, "zero") : "zero",
            BlendOp = payload.TryGetValue("blend_op", out var op) ? SafeString(op, "add") : "add",
            AlphaTest = ParseBool(payload.TryGetValue("alpha_test", out var alphaTest) ? alphaTest : null),
            AlphaFunc = payload.TryGetValue("alpha_func", out var alphaFunc) ? SafeString(alphaFunc, "greater") : "greater",
            AlphaRef = ParseFloat(payload.TryGetValue("alpha_ref", out var alphaRef) ? alphaRef : null),
            DepthWrite = ParseBool(payload.TryGetValue("depth_write", out var depthWrite) ? depthWrite : null, true),
            DepthTest = ParseBool(payload.TryGetValue("depth_test", out var depthTest) ? depthTest : null, true),
            CullMode = payload.TryGetValue("cull_mode", out var cull) ? SafeString(cull, "back") : "back",
            Emissive = ParseVec3(payload.TryGetValue("emissive", out var emissive) ? emissive : null),
            Specular = ParseVec3(payload.TryGetValue("specular", out var specular) ? specular : null),
            SpecularPower = ParseFloat(payload.TryGetValue("specular_power", out var power) ? power : null, 16f),
            DoubleSided = ParseBool(payload.TryGetValue("double_sided", out var doubleSided) ? doubleSided : null),
            Additive = ParseBool(payload.TryGetValue("additive", out var additive) ? additive : null),
            Transparent = ParseBool(payload.TryGetValue("transparent", out var transparent) ? transparent : null),
            Water = ParseBool(payload.TryGetValue("water", out var water) ? water : null),
            Lava = ParseBool(payload.TryGetValue("lava", out var lava) ? lava : null),
            NormalMap = NormalizeOptional(SafeString(payload.TryGetValue("normal_map", out var normal) ? normal : null, string.Empty)),
            ReceiveShadows = ParseBool(payload.TryGetValue("receive_shadows", out var shadows) ? shadows : null, true),
        };

        _states[state.Name] = state;
        var stem = Path.GetFileNameWithoutExtension(state.Name);
        if (!string.IsNullOrEmpty(stem) && !_states.ContainsKey(stem))
        {
            _aliases[stem] = state.Name;
        }
    }

    private static string SafeString(object? value, string fallback)
    {
        if (value is null)
        {
            return fallback;
        }
        var text = value.ToString();
        if (string.IsNullOrWhiteSpace(text))
        {
            return fallback;
        }
        return text.Trim();
    }

    private static string? NormalizeOptional(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? null : value;
    }

    private static bool ParseBool(object? value, bool defaultValue = false)
    {
        if (value is null)
        {
            return defaultValue;
        }
        if (value is bool b)
        {
            return b;
        }
        if (value is int i)
        {
            return i != 0;
        }
        if (value is JsonValue jsonValue)
        {
            if (jsonValue.TryGetValue(out bool boolValue))
            {
                return boolValue;
            }
            if (jsonValue.TryGetValue(out string? stringValue))
            {
                return ParseBool(stringValue, defaultValue);
            }
        }
        if (value is string text)
        {
            var lowered = text.Trim().ToLowerInvariant();
            return lowered switch
            {
                "1" or "true" or "yes" or "on" => true,
                "0" or "false" or "no" or "off" => false,
                _ => defaultValue,
            };
        }
        return defaultValue;
    }

    private static float ParseFloat(object? value, float defaultValue = 0f)
    {
        if (value is null)
        {
            return defaultValue;
        }
        if (value is float f)
        {
            return f;
        }
        if (value is double d)
        {
            return (float)d;
        }
        if (value is int i)
        {
            return i;
        }
        if (value is JsonValue jsonValue)
        {
            if (jsonValue.TryGetValue(out float floatValue))
            {
                return floatValue;
            }
            if (jsonValue.TryGetValue(out double doubleValue))
            {
                return (float)doubleValue;
            }
            if (jsonValue.TryGetValue(out string? stringValue))
            {
                return ParseFloat(stringValue, defaultValue);
            }
        }
        if (value is string text && float.TryParse(text, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var parsed))
        {
            return parsed;
        }
        return defaultValue;
    }

    private static (float, float, float) ParseVec3(object? value)
    {
        if (value is null)
        {
            return (0f, 0f, 0f);
        }
        if (value is JsonArray jsonArray && jsonArray.Count >= 3)
        {
            return (
                ParseFloat(jsonArray[0], 0f),
                ParseFloat(jsonArray[1], 0f),
                ParseFloat(jsonArray[2], 0f));
        }
        if (value is IEnumerable<object?> enumerable)
        {
            var parts = enumerable.Take(3).Select(v => ParseFloat(v, 0f)).ToArray();
            if (parts.Length == 3)
            {
                return (parts[0], parts[1], parts[2]);
            }
        }
        if (value is string text)
        {
            var tokens = text.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length >= 3)
            {
                return (
                    ParseFloat(tokens[0], 0f),
                    ParseFloat(tokens[1], 0f),
                    ParseFloat(tokens[2], 0f));
            }
        }
        return (0f, 0f, 0f);
    }

    private static string NormalizeKey(string textureName)
    {
        var text = textureName.Replace('\\', '/');
        foreach (var ext in new[] { ".ozj", ".ozt", ".jpg", ".jpeg", ".png", ".tga", ".bmp", ".dds" })
        {
            if (text.EndsWith(ext, StringComparison.OrdinalIgnoreCase))
            {
                text = text[..^ext.Length];
            }
        }
        return text.ToLowerInvariant();
    }

    private static MaterialState BuildFallback(string textureName)
    {
        var lowered = textureName.ToLowerInvariant();
        var state = new MaterialState
        {
            Name = NormalizeKey(textureName),
            Transparent = lowered.Contains("alpha") || lowered.Contains("glass"),
            Additive = lowered.Contains("glow") || lowered.Contains("flare"),
            Water = lowered.Contains("water") || lowered.Contains("river") || lowered.Contains("ocean"),
            Lava = lowered.Contains("lava") || lowered.Contains("magma"),
            ReceiveShadows = !lowered.Contains("shadow"),
        };
        if (state.Water)
        {
            state = state with
            {
                Transparent = true,
                DepthWrite = false,
                SrcBlend = "src_alpha",
                DstBlend = "one_minus_src_alpha",
            };
        }
        if (state.Lava)
        {
            state = state with
            {
                Transparent = true,
                Additive = true,
                Emissive = (0.6f, 0.3f, 0.1f),
            };
        }
        if (lowered.Contains("emissive") || lowered.Contains("light"))
        {
            state = state with { Emissive = (0.6f, 0.6f, 0.6f) };
        }
        return state;
    }
}
