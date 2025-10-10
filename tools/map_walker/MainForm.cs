using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using MapWalker.Rendering;
using MapWalker.Terrain;
using OpenTK.Mathematics;

namespace MapWalker;

public sealed class MainForm : Form
{
    private readonly TerrainRepository _repository = new();
    private readonly TerrainControl _view = new();
    private readonly TextBox _dataPathText = new() { Dock = DockStyle.Top };
    private readonly ComboBox _mapCombo = new() { Dock = DockStyle.Top, DropDownStyle = ComboBoxStyle.DropDownList };
    private readonly ComboBox _variantCombo = new() { Dock = DockStyle.Top, DropDownStyle = ComboBoxStyle.DropDownList };
    private readonly NumericUpDown _eyeHeight = new() { Minimum = 10, Maximum = 500, DecimalPlaces = 1, Increment = 1, Dock = DockStyle.Top, Value = 120 };
    private readonly NumericUpDown _speed = new() { Minimum = 100, Maximum = 5000, DecimalPlaces = 0, Increment = 50, Dock = DockStyle.Top, Value = 800 };
    private readonly NumericUpDown _colorR = new() { Minimum = 0, Maximum = 200, DecimalPlaces = 2, Increment = 0.05m, Dock = DockStyle.Top, Value = 100 };
    private readonly NumericUpDown _colorG = new() { Minimum = 0, Maximum = 200, DecimalPlaces = 2, Increment = 0.05m, Dock = DockStyle.Top, Value = 100 };
    private readonly NumericUpDown _colorB = new() { Minimum = 0, Maximum = 200, DecimalPlaces = 2, Increment = 0.05m, Dock = DockStyle.Top, Value = 100 };
    private readonly CheckBox _lockCheck = new() { Text = "Travar ao chão", Dock = DockStyle.Top, Checked = true };
    private readonly CheckBox _gridCheck = new() { Text = "Mostrar grade", Dock = DockStyle.Top, Checked = true };
    private readonly CheckBox _objectsCheck = new() { Text = "Mostrar objetos", Dock = DockStyle.Top, Checked = true };
    private readonly StatusStrip _statusStrip = new();
    private readonly ToolStripStatusLabel _cameraStatus = new("Câmera: -");
    private readonly ToolStripStatusLabel _tileStatus = new("Tile: -");
    private readonly Timer _statusTimer = new() { Interval = 200 };

    private IReadOnlyList<MapDescriptor> _maps = Array.Empty<MapDescriptor>();
    private bool _isUpdatingUi;

    public MainForm()
    {
        Text = "MU Map Walker";
        MinimumSize = new Size(1200, 800);

        var split = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Vertical,
            SplitterDistance = 320,
        };

        split.Panel1.Controls.Add(BuildControlPanel());
        split.Panel2.Controls.Add(_view);
        Controls.Add(split);

        _statusStrip.Items.Add(_cameraStatus);
        _statusStrip.Items.Add(new ToolStripStatusLabel { Spring = true });
        _statusStrip.Items.Add(_tileStatus);
        _statusStrip.Dock = DockStyle.Bottom;
        Controls.Add(_statusStrip);

        _variantCombo.Items.Add(new ComboBoxItem("Auto", AttributeVariant.Auto));
        _variantCombo.Items.Add(new ComboBoxItem("Padrão", AttributeVariant.Base));
        _variantCombo.Items.Add(new ComboBoxItem("Evento x10+1", AttributeVariant.Event1));
        _variantCombo.Items.Add(new ComboBoxItem("Evento x10+2", AttributeVariant.Event2));
        _variantCombo.SelectedIndex = 0;

        _mapCombo.SelectedIndexChanged += (_, _) => LoadSelectedMap();
        _variantCombo.SelectedIndexChanged += (_, _) => LoadSelectedMap();
        _lockCheck.CheckedChanged += (_, _) => SyncLock();
        _gridCheck.CheckedChanged += (_, _) => SyncGrid();
        _objectsCheck.CheckedChanged += (_, _) => SyncObjects();
        _eyeHeight.ValueChanged += (_, _) => _view.EyeHeight = (float)_eyeHeight.Value;
        _speed.ValueChanged += (_, _) => _view.MoveSpeed = (float)_speed.Value;
        _colorR.ValueChanged += (_, _) => UpdateBaseColor();
        _colorG.ValueChanged += (_, _) => UpdateBaseColor();
        _colorB.ValueChanged += (_, _) => UpdateBaseColor();

        _view.LockStateChanged += (_, locked) => UpdateCheck(_lockCheck, locked);
        _view.GridVisibilityChanged += (_, visible) => UpdateCheck(_gridCheck, visible);
        _view.ObjectVisibilityChanged += (_, visible) => UpdateCheck(_objectsCheck, visible);

        _statusTimer.Tick += (_, _) => UpdateStatus();
        _statusTimer.Start();
    }

    private Control BuildControlPanel()
    {
        var panel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 0,
            AutoScroll = true,
            Padding = new Padding(8),
        };

        panel.Controls.Add(BuildLabeledRow("Pasta de dados", BuildPathSelector()));
        panel.Controls.Add(BuildLabeledRow("Mapa", _mapCombo));
        panel.Controls.Add(BuildLabeledRow("Variante de atributo", _variantCombo));
        panel.Controls.Add(BuildLabeledRow("Altura do observador", _eyeHeight));
        panel.Controls.Add(BuildLabeledRow("Velocidade", _speed));
        panel.Controls.Add(BuildColorRow());
        panel.Controls.Add(_lockCheck);
        panel.Controls.Add(_gridCheck);
        panel.Controls.Add(_objectsCheck);

        var infoLabel = new Label
        {
            Dock = DockStyle.Top,
            Text = "Controles:\n- Clique direito: girar câmera\n- WASD: mover\n- Q/E: subir/descer (quando destravado)\n- Espaço: alternar travamento\n- G: alternar grade\n- O: alternar objetos",
            AutoSize = true,
            Padding = new Padding(0, 12, 0, 0),
        };
        panel.Controls.Add(infoLabel);
        panel.Controls.Add(new Panel { Dock = DockStyle.Fill });
        return panel;
    }

    private Control BuildPathSelector()
    {
        var container = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            FlowDirection = FlowDirection.LeftToRight,
            AutoSize = true,
            WrapContents = false,
        };

        _dataPathText.Width = 200;
        var browse = new Button { Text = "Escolher..." };
        browse.Click += (_, _) => BrowseForData();
        var apply = new Button { Text = "Carregar" };
        apply.Click += (_, _) => ApplyDataRoot(_dataPathText.Text);

        container.Controls.Add(_dataPathText);
        container.Controls.Add(browse);
        container.Controls.Add(apply);
        return container;
    }

    private Control BuildLabeledRow(string label, Control control)
    {
        var panel = new Panel { Dock = DockStyle.Top, AutoSize = true };
        var title = new Label { Text = label, Dock = DockStyle.Top, AutoSize = true };
        panel.Controls.Add(control);
        panel.Controls.Add(title);
        return panel;
    }

    private Control BuildColorRow()
    {
        var panel = new Panel { Dock = DockStyle.Top, AutoSize = true };
        var title = new Label { Text = "Cor base", Dock = DockStyle.Top, AutoSize = true };
        var row = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            FlowDirection = FlowDirection.LeftToRight,
            AutoSize = true,
            WrapContents = false,
        };

        _colorR.Width = 80;
        _colorG.Width = 80;
        _colorB.Width = 80;
        row.Controls.Add(new Label { Text = "R", AutoSize = true, TextAlign = ContentAlignment.MiddleLeft });
        row.Controls.Add(_colorR);
        row.Controls.Add(new Label { Text = "G", AutoSize = true, TextAlign = ContentAlignment.MiddleLeft });
        row.Controls.Add(_colorG);
        row.Controls.Add(new Label { Text = "B", AutoSize = true, TextAlign = ContentAlignment.MiddleLeft });
        row.Controls.Add(_colorB);

        panel.Controls.Add(row);
        panel.Controls.Add(title);
        return panel;
    }

    private void BrowseForData()
    {
        using var dialog = new FolderBrowserDialog
        {
            Description = "Selecione a pasta raiz dos dados do cliente",
            UseDescriptionForTitle = true,
        };

        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _dataPathText.Text = dialog.SelectedPath;
            ApplyDataRoot(dialog.SelectedPath);
        }
    }

    private void ApplyDataRoot(string path)
    {
        var resolved = _repository.SetDataRoot(path);
        if (resolved is null)
        {
            MessageBox.Show(this, "Pasta inválida. Certifique-se de apontar para a pasta que contém as pastas World.", "Erro", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        _dataPathText.Text = resolved.FullName;
        _maps = _repository.ListMaps();
        _mapCombo.Items.Clear();
        foreach (var descriptor in _maps)
        {
            _mapCombo.Items.Add($"[{descriptor.MapIndex:00}] {descriptor.DisplayName}");
        }

        if (_maps.Count > 0)
        {
            _mapCombo.SelectedIndex = 0;
        }
        else
        {
            _view.SetTerrain(null);
        }
    }

    private void LoadSelectedMap()
    {
        if (_mapCombo.SelectedIndex < 0 || _mapCombo.SelectedIndex >= _maps.Count)
        {
            return;
        }

        var descriptor = _maps[_mapCombo.SelectedIndex];
        var variant = ((ComboBoxItem)_variantCombo.SelectedItem!).Variant;
        try
        {
            Cursor = Cursors.WaitCursor;
            var data = _repository.LoadMap(descriptor, variant);
            _view.EyeHeight = (float)_eyeHeight.Value;
            _view.MoveSpeed = (float)_speed.Value;
            UpdateBaseColor();
            _view.SetTerrain(data);
            UpdateStatus();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Erro ao carregar mapa", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            Cursor = Cursors.Default;
        }
    }

    private void SyncLock()
    {
        if (_isUpdatingUi)
        {
            return;
        }

        _view.LockToGround = _lockCheck.Checked;
    }

    private void SyncGrid()
    {
        if (_isUpdatingUi)
        {
            return;
        }

        _view.ShowGrid = _gridCheck.Checked;
    }

    private void SyncObjects()
    {
        if (_isUpdatingUi)
        {
            return;
        }

        _view.ShowObjects = _objectsCheck.Checked;
    }

    private void UpdateCheck(CheckBox box, bool value)
    {
        _isUpdatingUi = true;
        try
        {
            box.Checked = value;
        }
        finally
        {
            _isUpdatingUi = false;
        }
    }

    private void UpdateBaseColor()
    {
        _view.BaseColor = new Vector3((float)(_colorR.Value / 100m), (float)(_colorG.Value / 100m), (float)(_colorB.Value / 100m));
    }

    private void UpdateStatus()
    {
        var terrain = _view.Terrain;
        if (terrain is null)
        {
            _cameraStatus.Text = "Câmera: -";
            _tileStatus.Text = "Tile: -";
            return;
        }

        var pos = _view.CameraPosition;
        _cameraStatus.Text = $"Câmera: {pos.X:0.0}, {pos.Y:0.0}, {pos.Z:0.0}";
        var tile = terrain.TileIndicesAt(pos.X, pos.Z);
        if (tile is null)
        {
            _tileStatus.Text = "Tile: -";
        }
        else
        {
            var (layer1, layer2, alpha) = tile.Value;
            _tileStatus.Text = $"Tile: L1={layer1} L2={layer2} α={alpha:0.00}";
        }
    }

    private sealed record ComboBoxItem(string Text, AttributeVariant Variant)
    {
        public override string ToString() => Text;
    }
}
