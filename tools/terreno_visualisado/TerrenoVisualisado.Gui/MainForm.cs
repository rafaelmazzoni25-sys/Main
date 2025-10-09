using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using TerrenoVisualisado.Core;

namespace TerrenoVisualisado.Gui;

public class MainForm : Form
{
    private readonly TextBox _rootPath;
    private readonly ComboBox _worldSelector;
    private readonly TextBox _objectPath;
    private readonly TextBox _enumPath;
    private readonly CheckBox _forceMapId;
    private readonly NumericUpDown _mapIdNumeric;
    private readonly CheckBox _customHeightScale;
    private readonly NumericUpDown _heightScaleNumeric;
    private readonly CheckBox _forceExtendedHeight;
    private readonly Button _loadButton;
    private readonly Button _exportButton;
    private readonly ComboBox _previewMode;
    private readonly CheckBox _overlayObjects;
    private readonly TabControl _previewTabs;
    private readonly PictureBox _preview2D;
    private readonly TerrainGlViewer _preview3D;
    private readonly TextBox _summary;
    private readonly ListView _objectList;

    private readonly WorldLoader _loader = new();
    private WorldData? _world;
    private readonly List<WorldEntry> _worldEntries = new();
    private WorldEntry? _selectedWorld;

    public MainForm()
    {
        Text = "Terreno Visualisado";
        MinimumSize = new System.Drawing.Size(960, 720);

        var layout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 2,
        };
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        Controls.Add(layout);

        var controlPanel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 16,
            AutoSize = true,
        };
        controlPanel.ColumnStyles.Clear();
        for (var i = 0; i < controlPanel.ColumnCount; i++)
        {
            controlPanel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        }

        _rootPath = new TextBox { Width = 260, Anchor = AnchorStyles.Left | AnchorStyles.Right };
        _worldSelector = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 220 };
        _worldSelector.SelectedIndexChanged += (_, _) => OnWorldSelected();
        _objectPath = new TextBox { Width = 200, Anchor = AnchorStyles.Left | AnchorStyles.Right };
        _enumPath = new TextBox { Width = 200, Anchor = AnchorStyles.Left | AnchorStyles.Right };

        var rootLabel = new Label { Text = "Raiz:", Anchor = AnchorStyles.Left, AutoSize = true };
        var worldLabel = new Label { Text = "Mapa:", Anchor = AnchorStyles.Left, AutoSize = true };
        var objectLabel = new Label { Text = "Objetos:", Anchor = AnchorStyles.Left, AutoSize = true };
        var enumLabel = new Label { Text = "Enum:", Anchor = AnchorStyles.Left, AutoSize = true };

        var browseRoot = new Button { Text = "...", AutoSize = true };
        browseRoot.Click += (_, _) => BrowseRoot();
        var refreshWorlds = new Button { Text = "Atualizar", AutoSize = true };
        refreshWorlds.Click += (_, _) => PopulateWorldsFromRoot();
        var browseObjects = new Button { Text = "...", AutoSize = true };
        browseObjects.Click += (_, _) => BrowseFolder(_objectPath);
        var browseEnum = new Button { Text = "...", AutoSize = true };
        browseEnum.Click += (_, _) => BrowseFile(_enumPath, "EnumModelType.eum|*.eum|Todos|*.*");

        _forceMapId = new CheckBox { Text = "Forçar mapa:", AutoSize = true };
        _mapIdNumeric = new NumericUpDown
        {
            Minimum = 0,
            Maximum = 999,
            Width = 60,
            Enabled = false,
        };
        _forceMapId.CheckedChanged += (_, _) => _mapIdNumeric.Enabled = _forceMapId.Checked;

        _customHeightScale = new CheckBox { Text = "Altura x", AutoSize = true };
        _heightScaleNumeric = new NumericUpDown
        {
            Minimum = 0,
            Maximum = 100,
            DecimalPlaces = 2,
            Increment = 0.1M,
            Value = 1.50M,
            Width = 70,
            Enabled = false,
        };
        _customHeightScale.CheckedChanged += (_, _) => _heightScaleNumeric.Enabled = _customHeightScale.Checked;

        _forceExtendedHeight = new CheckBox { Text = "Forçar TerrainHeightNew", AutoSize = true };

        _loadButton = new Button { Text = "Carregar", AutoSize = true };
        _loadButton.Click += (_, _) => LoadWorld();

        _exportButton = new Button { Text = "Exportar JSON", AutoSize = true, Enabled = false };
        _exportButton.Click += (_, _) => ExportJson();

        _previewMode = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 150 };
        _previewMode.Items.AddRange(new object[]
        {
            new PreviewItem("Altura", PreviewMode.Height),
            new PreviewItem("Layer 1", PreviewMode.Layer1),
            new PreviewItem("Layer 2", PreviewMode.Layer2),
            new PreviewItem("Alfa", PreviewMode.Alpha),
            new PreviewItem("Atributos", PreviewMode.Attributes),
        });
        _previewMode.SelectedIndex = 0;
        _previewMode.SelectedIndexChanged += (_, _) => RenderPreview();

        _overlayObjects = new CheckBox { Text = "Sobrepor objetos", AutoSize = true, Checked = true };
        _overlayObjects.CheckedChanged += (_, _) => RenderPreview();

        controlPanel.Controls.Add(rootLabel, 0, 0);
        controlPanel.Controls.Add(_rootPath, 1, 0);
        controlPanel.SetColumnSpan(_rootPath, 3);
        controlPanel.Controls.Add(browseRoot, 4, 0);
        controlPanel.Controls.Add(refreshWorlds, 5, 0);
        controlPanel.Controls.Add(worldLabel, 6, 0);
        controlPanel.Controls.Add(_worldSelector, 7, 0);
        controlPanel.SetColumnSpan(_worldSelector, 3);
        controlPanel.Controls.Add(objectLabel, 10, 0);
        controlPanel.Controls.Add(_objectPath, 11, 0);
        controlPanel.SetColumnSpan(_objectPath, 2);
        controlPanel.Controls.Add(browseObjects, 13, 0);

        controlPanel.Controls.Add(enumLabel, 0, 1);
        controlPanel.Controls.Add(_enumPath, 1, 1);
        controlPanel.SetColumnSpan(_enumPath, 3);
        controlPanel.Controls.Add(browseEnum, 4, 1);
        controlPanel.Controls.Add(_forceMapId, 5, 1);
        controlPanel.Controls.Add(_mapIdNumeric, 6, 1);
        controlPanel.Controls.Add(_customHeightScale, 7, 1);
        controlPanel.Controls.Add(_heightScaleNumeric, 8, 1);
        controlPanel.Controls.Add(_forceExtendedHeight, 9, 1);

        var controlRow2 = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            AutoSize = true,
        };
        controlRow2.Controls.Add(_previewMode);
        controlRow2.Controls.Add(_overlayObjects);
        controlRow2.Controls.Add(_loadButton);
        controlRow2.Controls.Add(_exportButton);

        layout.Controls.Add(controlPanel, 0, 0);
        layout.Controls.Add(controlRow2, 0, 1);

        var split = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Vertical,
            SplitterDistance = 650,
        };
        layout.Controls.Add(split, 0, 2);

        _previewTabs = new TabControl { Dock = DockStyle.Fill };
        _preview2D = new PictureBox
        {
            Dock = DockStyle.Fill,
            SizeMode = PictureBoxSizeMode.Zoom,
            BackColor = System.Drawing.Color.Black,
        };
        var preview2DPage = new TabPage("Pré-visualização 2D") { Padding = new Padding(3) };
        preview2DPage.Controls.Add(_preview2D);

        _preview3D = new TerrainGlViewer();
        var preview3DPage = new TabPage("Visualização 3D") { Padding = new Padding(3) };
        preview3DPage.Controls.Add(_preview3D);

        _previewTabs.TabPages.Add(preview2DPage);
        _previewTabs.TabPages.Add(preview3DPage);

        split.Panel1.Controls.Add(_previewTabs);

        var tabs = new TabControl { Dock = DockStyle.Fill };
        split.Panel2.Controls.Add(tabs);

        _summary = new TextBox
        {
            Dock = DockStyle.Fill,
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Vertical,
        };
        var summaryPage = new TabPage("Resumo") { Padding = new Padding(3) };
        summaryPage.Controls.Add(_summary);

        _objectList = new ListView
        {
            Dock = DockStyle.Fill,
            View = View.Details,
            FullRowSelect = true,
        };
        _objectList.Columns.Add("Objeto", 180);
        _objectList.Columns.Add("Quantidade", 90);
        var objectsPage = new TabPage("Objetos") { Padding = new Padding(3) };
        objectsPage.Controls.Add(_objectList);

        tabs.TabPages.Add(summaryPage);
        tabs.TabPages.Add(objectsPage);

        // Adjust layout row styles
        layout.RowCount = 3;
        layout.RowStyles[0] = new RowStyle(SizeType.AutoSize);
        layout.RowStyles[1] = new RowStyle(SizeType.AutoSize);
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _preview2D.Image?.Dispose();
        }
        base.Dispose(disposing);
    }

    private void BrowseFolder(TextBox target)
    {
        using var dialog = new FolderBrowserDialog();
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            target.Text = dialog.SelectedPath;
        }
    }

    private void BrowseRoot()
    {
        using var dialog = new FolderBrowserDialog();
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _rootPath.Text = dialog.SelectedPath;
            PopulateWorldsFromRoot();
        }
    }

    private void BrowseFile(TextBox target, string filter)
    {
        using var dialog = new OpenFileDialog { Filter = filter };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            target.Text = dialog.FileName;
        }
    }

    private void LoadWorld()
    {
        if (_selectedWorld is null)
        {
            MessageBox.Show(this, "Selecione um mapa disponível.", "Terreno Visualisado", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        var options = new WorldLoader.LoadOptions
        {
            ObjectRoot = string.IsNullOrWhiteSpace(_objectPath.Text) ? null : _objectPath.Text,
            EnumPath = string.IsNullOrWhiteSpace(_enumPath.Text) ? null : _enumPath.Text,
            ForceExtendedHeight = _forceExtendedHeight.Checked,
            HeightScale = _customHeightScale.Checked ? (float?)Convert.ToDouble(_heightScaleNumeric.Value) : null,
            MapId = _forceMapId.Checked ? (int?)Convert.ToInt32(_mapIdNumeric.Value) : null,
        };

        try
        {
            Cursor = Cursors.WaitCursor;
            _world = _loader.Load(_selectedWorld.Path, options);
            _exportButton.Enabled = true;
            UpdateSummary();
            RenderPreview();
            _preview3D.DisplayWorld(_world);
        }
        catch (Exception ex)
        {
            _preview3D.DisplayWorld(null);
            MessageBox.Show(this, ex.Message, "Erro ao carregar", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            Cursor = Cursors.Default;
        }
    }

    private void ExportJson()
    {
        if (_world is null)
        {
            return;
        }

        using var dialog = new SaveFileDialog
        {
            Filter = "JSON|*.json|Todos|*.*",
            FileName = $"terrainsummary_{_world.MapId}.json",
        };

        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            try
            {
                WorldExporter.WriteJson(_world, dialog.FileName);
                MessageBox.Show(this, "Exportação concluída!", "Terreno Visualisado", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "Erro ao exportar", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }

    private void UpdateSummary()
    {
        if (_world is null)
        {
            _summary.Text = string.Empty;
            _objectList.Items.Clear();
            return;
        }

        var builder = new StringBuilder();
        builder.AppendLine($"Mapa: {_world.MapId}");
        builder.AppendLine($"World: {_world.WorldPath}");
        builder.AppendLine($"EncTerrain: {_world.ObjectsPath}");
        builder.AppendLine($"Object dir: {_world.ObjectDirectory}");
        builder.AppendLine($"Objetos: {_world.Objects.Count} (versão {_world.ObjectVersion})");

        var tileCounts = new Dictionary<byte, int>();
        for (var i = 0; i < _world.Terrain.Layer1.Length; i++)
        {
            var tile1 = _world.Terrain.Layer1[i];
            var tile2 = _world.Terrain.Layer2[i];
            tileCounts.TryGetValue(tile1, out var count1);
            tileCounts[tile1] = count1 + 1;
            tileCounts.TryGetValue(tile2, out var count2);
            tileCounts[tile2] = count2 + 1;
        }

        builder.AppendLine("Tiles mais comuns:");
        foreach (var pair in tileCounts.OrderByDescending(kv => kv.Value).Take(8))
        {
            builder.AppendLine($"  {pair.Key:D3}: {pair.Value}");
        }

        var attributeCounts = new Dictionary<ushort, int>();
        foreach (var attr in _world.Terrain.Attributes)
        {
            attributeCounts.TryGetValue(attr, out var count);
            attributeCounts[attr] = count + 1;
        }
        builder.AppendLine("Atributos mais comuns:");
        foreach (var pair in attributeCounts.OrderByDescending(kv => kv.Value).Take(8))
        {
            builder.AppendLine($"  0x{pair.Key:X3}: {pair.Value}");
        }

        _summary.Text = builder.ToString();

        var objectCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var obj in _world.Objects)
        {
            var key = obj.TypeName ?? $"ID_{obj.TypeId}";
            objectCounts.TryGetValue(key, out var count);
            objectCounts[key] = count + 1;
        }

        _objectList.BeginUpdate();
        _objectList.Items.Clear();
        foreach (var pair in objectCounts.OrderByDescending(kv => kv.Value))
        {
            var item = new ListViewItem(pair.Key) { SubItems = { pair.Value.ToString()! } };
            _objectList.Items.Add(item);
        }
        _objectList.EndUpdate();
    }

    private void RenderPreview()
    {
        if (_world is null)
        {
            _preview2D.Image?.Dispose();
            _preview2D.Image = null;
            _preview3D.DisplayWorld(null);
            return;
        }

        if (_previewMode.SelectedItem is not PreviewItem item)
        {
            return;
        }

        _preview2D.Image?.Dispose();
        _preview2D.Image = TerrainPreviewRenderer.Render(_world, item.Mode, _overlayObjects.Checked);
    }

    private void PopulateWorldsFromRoot()
    {
        _worldEntries.Clear();
        _worldSelector.Items.Clear();
        _selectedWorld = null;

        var root = _rootPath.Text;
        if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root))
        {
            return;
        }

        try
        {
            foreach (var entry in EnumerateWorlds(root))
            {
                _worldEntries.Add(entry);
                _worldSelector.Items.Add(entry);
            }

            if (_worldEntries.Count > 0)
            {
                _worldSelector.SelectedIndex = 0;
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Erro ao listar mundos", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private void OnWorldSelected()
    {
        if (_worldSelector.SelectedItem is not WorldEntry entry)
        {
            _selectedWorld = null;
            return;
        }

        _selectedWorld = entry;

        if (!_forceMapId.Checked && entry.MapId.HasValue)
        {
            var mapId = entry.MapId.Value;
            mapId = Math.Clamp(mapId, (int)_mapIdNumeric.Minimum, (int)_mapIdNumeric.Maximum);
            _mapIdNumeric.Value = mapId;
        }

        var guess = GuessObjectDirectory(entry.Path);
        if (guess is not null && !string.Equals(_objectPath.Text, guess, StringComparison.OrdinalIgnoreCase))
        {
            _objectPath.Text = guess;
        }
    }

    private static IEnumerable<WorldEntry> EnumerateWorlds(string root)
    {
        var directories = Enumerable.Empty<string>();
        try
        {
            directories = Directory.EnumerateDirectories(root);
        }
        catch (UnauthorizedAccessException)
        {
            yield break;
        }
        catch (IOException)
        {
            yield break;
        }

        foreach (var path in directories.OrderBy(p => p, StringComparer.OrdinalIgnoreCase))
        {
            var info = new DirectoryInfo(path);
            if (!info.Name.StartsWith("World", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (!HasTerrainFiles(info.FullName))
            {
                continue;
            }

            yield return new WorldEntry(info.Name, info.FullName, ExtractDigits(info.Name));
        }
    }

    private static bool HasTerrainFiles(string directory)
    {
        try
        {
            return Directory.EnumerateFiles(directory, "EncTerrain*.map", SearchOption.TopDirectoryOnly).Any();
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
        catch (IOException)
        {
            return false;
        }
    }

    private static string? GuessObjectDirectory(string worldPath)
    {
        var info = new DirectoryInfo(worldPath);
        var name = info.Name;
        if (!name.StartsWith("World", StringComparison.OrdinalIgnoreCase) || name.Length <= 5)
        {
            return null;
        }

        var suffix = name[5..];
        var parent = info.Parent?.FullName ?? info.FullName;
        var candidate = Path.Combine(parent, "Object" + suffix);
        if (Directory.Exists(candidate))
        {
            return candidate;
        }

        var lowerCandidate = Path.Combine(parent, "object" + suffix);
        if (Directory.Exists(lowerCandidate))
        {
            return lowerCandidate;
        }

        return null;
    }

    private static int? ExtractDigits(string text)
    {
        var digits = new string(text.Where(char.IsDigit).ToArray());
        if (digits.Length == 0)
        {
            return null;
        }

        return int.TryParse(digits, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
            ? value
            : null;
    }

    private sealed record PreviewItem(string Text, PreviewMode Mode)
    {
        public override string ToString() => Text;
    }

    private sealed record WorldEntry(string Name, string Path, int? MapId)
    {
        public override string ToString()
        {
            return MapId.HasValue ? $"{Name} (Mapa {MapId.Value})" : Name;
        }
    }
}
