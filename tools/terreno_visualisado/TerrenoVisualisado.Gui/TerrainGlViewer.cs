using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Windows.Forms;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using OpenTK.WinForms;
using OpenTK.Windowing.Common;
using TerrenoVisualisado.Core;

namespace TerrenoVisualisado.Gui;

internal sealed class TerrainGlViewer : UserControl
{
    private static readonly Vector3 DefaultLightDirection = LightingProfile.Default.Direction;
    private static readonly Vector3 DefaultFogColor = new(0.23f, 0.28f, 0.34f);
    private static readonly Vector2 DefaultFogParams = new(0.00045f, 1.35f);

    private readonly GLControl? _glControl;
    private readonly Control _renderSurface;
    private readonly OrbitCamera _camera = new();
    private readonly TerrainRenderer3D _renderer = new();
    private readonly ObjectRenderer3D _objectRenderer = new();
    private readonly SkyRenderer _skyRenderer = new();
    private readonly System.Windows.Forms.Timer _inputTimer;
    private readonly Stopwatch _deltaWatch = Stopwatch.StartNew();
    private readonly HashSet<Keys> _keys = new();
    private readonly ToolTip _toolTip = new();
    private readonly CheckBox _terrainToggle;
    private readonly CheckBox _objectToggle;
    private readonly CheckBox _fogToggle;
    private readonly CheckBox _skyToggle;
    private readonly CheckBox _lightingToggle;
    private readonly CheckBox _animationToggle;
    private readonly CheckBox _gameModeToggle;
    private bool _showTerrain = true;
    private bool _showObjects = true;
    private bool _fogEnabled = true;
    private bool _skyEnabled = true;
    private bool _lightingEnabled = true;
    private bool _animationsEnabled = true;
    private TerrainMesh? _mesh;
    private LightingProfile _lightingProfile = LightingProfile.Default;
    private bool _contextReady;
    private bool _renderingUnavailable;
    private string? _renderingError;
    private bool _renderingErrorShown;
    private bool _rotating;
    private bool _panning;
    private Point _lastMouse;
    private bool _flightMode;
    private bool _suppressGameModeEvent;
    private float _flightYaw;
    private float _flightPitch;
    private float _flightSpeed = 1000f;
    private Vector3 _flightPosition;

    public TerrainGlViewer()
    {
        Dock = DockStyle.Fill;
        DoubleBuffered = true;

        try
        {
            _glControl = new GLControl(new GLControlSettings
            {
                API = ContextAPI.OpenGL,
                APIVersion = new Version(3, 3),
                Profile = ContextProfile.Core,
                Flags = ContextFlags.Default,
            })
            {
                Dock = DockStyle.Fill,
                BackColor = Color.Black,
                TabStop = true,
            };
            _renderSurface = _glControl;
        }
        catch (Exception ex)
        {
            _renderingUnavailable = true;
            _renderingError = $"Não foi possível inicializar a visualização 3D: {ex.Message}";
            _renderSurface = CreateFallbackSurface();
            ScheduleRenderingErrorMessage();
        }

        var layout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 2,
        };
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));
        Controls.Add(layout);

        var optionsPanel = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink,
            WrapContents = false,
            Padding = new Padding(8, 4, 8, 4),
            Margin = new Padding(0),
        };

        _terrainToggle = CreateToggle("Terreno", (_, _) =>
        {
            _showTerrain = _terrainToggle.Checked;
            _glControl?.Invalidate();
        }, true);
        _objectToggle = CreateToggle("Objetos", (_, _) =>
        {
            _showObjects = _objectToggle.Checked;
            _glControl?.Invalidate();
        }, true);
        _fogToggle = CreateToggle("Névoa", (_, _) =>
        {
            _fogEnabled = _fogToggle.Checked;
            ApplyRenderingSettings();
        }, true);
        _skyToggle = CreateToggle("Céu", (_, _) =>
        {
            _skyEnabled = _skyToggle.Checked;
            _glControl?.Invalidate();
        }, true);
        _lightingToggle = CreateToggle("Iluminação", (_, _) =>
        {
            _lightingEnabled = _lightingToggle.Checked;
            ApplyRenderingSettings();
        }, true);
        _animationToggle = CreateToggle("Animações", (_, _) =>
        {
            _animationsEnabled = _animationToggle.Checked;
            ApplyRenderingSettings();
        }, true);
        _gameModeToggle = CreateToggle("Modo jogo (F)", GameModeToggleChanged, false);

        optionsPanel.Controls.AddRange(new Control[]
        {
            _terrainToggle,
            _objectToggle,
            _fogToggle,
            _skyToggle,
            _lightingToggle,
            _animationToggle,
            _gameModeToggle,
        });

        layout.Controls.Add(optionsPanel, 0, 0);
        _renderSurface.Margin = new Padding(0);
        layout.Controls.Add(_renderSurface, 0, 1);

        _toolTip.SetToolTip(_terrainToggle, "Exibe ou oculta a malha do terreno.");
        _toolTip.SetToolTip(_objectToggle, "Instancia os modelos BMD carregados.");
        _toolTip.SetToolTip(_fogToggle, "Alterna a névoa de distância.");
        _toolTip.SetToolTip(_skyToggle, "Alterna o gradiente de céu.");
        _toolTip.SetToolTip(_lightingToggle, "Aplica iluminação difusa e especular.");
        _toolTip.SetToolTip(_animationToggle, "Reproduz animações dos objetos.");
        _toolTip.SetToolTip(_gameModeToggle, "Ativa o modo de navegação livre com câmera em primeira pessoa.");

        if (_glControl is not null)
        {
            _toolTip.SetToolTip(_glControl,
                "Modo órbita: botão esquerdo para orbitar, direito para pan, roda para zoom.\n" +
                "Modo jogo: marque a caixa ou pressione F. Use WASD para mover, E/Espaço para subir, Q para descer, " +
                "Shift acelera e Ctrl desacelera.");

            _skyRenderer.Configure(DefaultFogColor, _lightingProfile);

            _glControl.Load += HandleLoad;
            _glControl.Resize += HandleResize;
            _glControl.Paint += HandlePaint;
            _glControl.MouseDown += HandleMouseDown;
            _glControl.MouseUp += HandleMouseUp;
            _glControl.MouseMove += HandleMouseMove;
            _glControl.MouseWheel += HandleMouseWheel;
            _glControl.KeyDown += HandleKeyDown;
            _glControl.KeyUp += HandleKeyUp;
            _glControl.LostFocus += HandleLostFocus;
            _glControl.PreviewKeyDown += HandlePreviewKeyDown;
        }
        else
        {
            optionsPanel.Enabled = false;
        }

        _inputTimer = new System.Windows.Forms.Timer
        {
            Interval = 16,
        };
        _inputTimer.Tick += HandleTick;
        _inputTimer.Start();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _inputTimer.Stop();
            _inputTimer.Dispose();
            if (_glControl is not null)
            {
                try
                {
                    if (_contextReady && !_renderingUnavailable)
                    {
                        _glControl.MakeCurrent();
                    }
                }
                catch
                {
                    // Ignore errors when restoring the OpenGL context during disposal.
                }
            }

            try
            {
                _renderer.Dispose();
            }
            catch
            {
            }

            try
            {
                _objectRenderer.Dispose();
            }
            catch
            {
            }

            try
            {
                _skyRenderer.Dispose();
            }
            catch
            {
            }
            _toolTip.Dispose();
            _glControl?.Dispose();
        }
        base.Dispose(disposing);
    }

    public void DisplayWorld(WorldData? world)
    {
        if (_renderingUnavailable)
        {
            return;
        }

        if (world is null)
        {
            _mesh = null;
            _lightingProfile = LightingProfile.Default;
            _renderer.UpdateData(null, null, null, _lightingProfile, DefaultFogColor, DefaultFogParams, _fogEnabled, _lightingEnabled);
            _objectRenderer.UpdateWorld(null);
            _objectRenderer.ConfigureEnvironment(_lightingProfile, DefaultFogColor, DefaultFogParams, _fogEnabled, _lightingEnabled);
            _skyRenderer.Configure(DefaultFogColor, _lightingProfile);
            _flightMode = false;
            _rotating = false;
            _panning = false;
            UpdateGameModeToggle();
            ApplyRenderingSettings();
            if (_contextReady)
            {
                try
                {
                    _glControl.MakeCurrent();
                    _renderer.EnsureResources();
                    _objectRenderer.EnsureResources();
                    _glControl.Invalidate();
                }
                catch (Exception ex)
                {
                    HandleRenderingFailure("inicializar recursos 3D", ex);
                }
            }
            return;
        }

        var materialFlags = world.Visual?.MaterialFlagsPerTile;
        _mesh = TerrainMeshBuilder.Build(world.Terrain, materialFlags);
        var context = MapContext.ForMapId(world.MapId);
        var lightDirection = context.IsBattleCastle
            ? new Vector3(0.5f, -1.0f, 1.0f)
            : DefaultLightDirection;
        var terrainTexture = world.Visual?.LitCompositeTexture ?? world.Visual?.CompositeTexture;
        var lightMap = world.Visual?.LightMap;
        var fogColor = EstimateFogColor(terrainTexture ?? world.Visual?.CompositeTexture, context);
        var fogParams = EstimateFogParams(context);
        _lightingProfile = LightingProfile.Create(lightDirection, fogColor, context);
        _renderer.UpdateData(_mesh, terrainTexture, lightMap, _lightingProfile, fogColor, fogParams, _fogEnabled, _lightingEnabled);
        _objectRenderer.UpdateWorld(world);
        _objectRenderer.ConfigureEnvironment(_lightingProfile, fogColor, fogParams, _fogEnabled, _lightingEnabled);
        _skyRenderer.Configure(fogColor, _lightingProfile);
        ResetCamera();

        if (_contextReady)
        {
            try
            {
                _glControl.MakeCurrent();
                _renderer.EnsureResources();
                _objectRenderer.EnsureResources();
                _glControl.Invalidate();
            }
            catch (Exception ex)
            {
                HandleRenderingFailure("atualizar recursos 3D", ex);
            }
        }

        ApplyRenderingSettings();
    }

    private void ResetCamera()
    {
        if (_mesh is null)
        {
            return;
        }

        var extent = (WorldLoader.TerrainSize - 1) * WorldLoader.TerrainScale;
        var center = new Vector3(extent * 0.5f, extent * 0.5f, (_mesh.BoundsMin.Z + _mesh.BoundsMax.Z) * 0.5f);
        _camera.Target = center;
        _camera.Azimuth = MathHelper.DegreesToRadians(45f);
        _camera.Elevation = MathHelper.DegreesToRadians(35f);
        var radius = Math.Max(extent, _mesh.BoundsMax.Z - _mesh.BoundsMin.Z);
        _camera.Distance = Math.Max(1000f, radius * 1.2f);
        _flightMode = false;
        _rotating = false;
        _panning = false;
        SyncFlightFromOrbit();
        UpdateGameModeToggle();
        _glControl.Cursor = Cursors.Default;
    }

    private void HandleLoad(object? sender, EventArgs e)
    {
        if (_renderingUnavailable)
        {
            return;
        }

        _contextReady = true;
        try
        {
            _glControl.MakeCurrent();
            GL.ClearColor(0.1f, 0.14f, 0.2f, 1f);
            GL.Enable(EnableCap.DepthTest);
            _renderer.EnsureResources();
            _objectRenderer.EnsureResources();
        }
        catch (Exception ex)
        {
            HandleRenderingFailure("inicializar a visualização 3D", ex);
        }
    }

    private void HandleResize(object? sender, EventArgs e)
    {
        if (_renderingUnavailable)
        {
            return;
        }

        if (!_contextReady)
        {
            return;
        }

        try
        {
            _glControl.MakeCurrent();
            GL.Viewport(0, 0, Math.Max(1, _glControl.Width), Math.Max(1, _glControl.Height));
            _glControl.Invalidate();
        }
        catch (Exception ex)
        {
            HandleRenderingFailure("ajustar o viewport 3D", ex);
        }
    }

    private void HandlePaint(object? sender, PaintEventArgs e)
    {
        if (_renderingUnavailable)
        {
            DrawFallbackMessage(e.Graphics);
            return;
        }

        if (!_contextReady)
        {
            e.Graphics.Clear(Color.Black);
            return;
        }

        try
        {
            _glControl.MakeCurrent();
            var clearColor = _skyEnabled ? _skyRenderer.BottomColor : new Vector3(0.05f, 0.07f, 0.11f);
            GL.ClearColor(clearColor.X, clearColor.Y, clearColor.Z, 1f);
            GL.Viewport(0, 0, Math.Max(1, _glControl.Width), Math.Max(1, _glControl.Height));
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            if (_skyEnabled)
            {
                _skyRenderer.EnsureResources();
                _skyRenderer.Render();
            }

            if (_mesh is null)
            {
                _glControl.SwapBuffers();
                return;
            }

            var aspect = _glControl.Width / (float)Math.Max(1, _glControl.Height);
            var view = _flightMode ? GetFlightViewMatrix() : _camera.ViewMatrix;
            var projection = _camera.ProjectionMatrix(aspect);
            var cameraPosition = _flightMode ? _flightPosition : _camera.Position;

            if (_showTerrain)
            {
                _renderer.EnsureResources();
                _renderer.Render(view, projection, cameraPosition);
            }

            if (_showObjects)
            {
                _objectRenderer.EnsureResources();
                _objectRenderer.Render(view, projection, cameraPosition);
            }

            _glControl.SwapBuffers();
        }
        catch (Exception ex)
        {
            HandleRenderingFailure("renderizar a cena 3D", ex);
            DrawFallbackMessage(e.Graphics);
        }
    }

    private void HandleMouseDown(object? sender, MouseEventArgs e)
    {
        if (_renderingUnavailable)
        {
            return;
        }

        _glControl.Focus();
        if (e.Button == MouseButtons.Left)
        {
            _rotating = true;
        }
        else if (e.Button == MouseButtons.Right && !_flightMode)
        {
            _panning = true;
        }
        _lastMouse = e.Location;
    }

    private void HandleMouseUp(object? sender, MouseEventArgs e)
    {
        if (_renderingUnavailable)
        {
            return;
        }

        if (e.Button == MouseButtons.Left)
        {
            _rotating = false;
        }
        else if (e.Button == MouseButtons.Right)
        {
            _panning = false;
        }
    }

    private void HandleMouseMove(object? sender, MouseEventArgs e)
    {
        if (_renderingUnavailable)
        {
            return;
        }

        if (!_rotating && !_panning)
        {
            return;
        }

        var deltaX = e.X - _lastMouse.X;
        var deltaY = e.Y - _lastMouse.Y;
        _lastMouse = e.Location;

        if (_flightMode && _rotating)
        {
            _flightYaw -= deltaX * 0.01f;
            var newPitch = _flightPitch - deltaY * 0.01f;
            _flightPitch = Math.Clamp(newPitch, -1.4f, 1.4f);
        }
        else if (_rotating)
        {
            _camera.Azimuth -= deltaX * 0.01f;
            var newElevation = _camera.Elevation - deltaY * 0.01f;
            _camera.Elevation = Math.Clamp(newElevation, -1.2f, 1.2f);
        }
        else if (_panning)
        {
            var right = _camera.Right;
            var up = _camera.Up;
            var scale = _camera.Distance * 0.0025f;
            _camera.Target -= right * deltaX * scale;
            _camera.Target += up * deltaY * scale;
        }

        _glControl.Invalidate();
    }

    private void HandleMouseWheel(object? sender, MouseEventArgs e)
    {
        if (_renderingUnavailable)
        {
            return;
        }

        var factor = 1f - Math.Sign(e.Delta) * 0.1f;
        if (_flightMode)
        {
            _flightSpeed = Math.Clamp(_flightSpeed * factor, 50f, 200000f);
        }
        else
        {
            _camera.Distance = Math.Clamp(_camera.Distance * factor, 200f, 200000f);
        }
        _glControl.Invalidate();
    }

    private void HandleKeyDown(object? sender, KeyEventArgs e)
    {
        if (_renderingUnavailable)
        {
            return;
        }

        _keys.Add(e.KeyCode);
        if (e.KeyCode == Keys.F)
        {
            SetFlightMode(!_flightMode);
            e.Handled = true;
        }
        else if (e.KeyCode == Keys.R)
        {
            ResetCamera();
            if (_flightMode)
            {
                SyncFlightFromOrbit();
            }
            UpdateGameModeToggle();
            e.Handled = true;
        }
    }

    private void HandleKeyUp(object? sender, KeyEventArgs e)
    {
        if (_renderingUnavailable)
        {
            return;
        }

        _keys.Remove(e.KeyCode);
    }

    private void HandleLostFocus(object? sender, EventArgs e)
    {
        if (_renderingUnavailable)
        {
            return;
        }

        _keys.Clear();
        _rotating = false;
        _panning = false;
    }

    private void HandlePreviewKeyDown(object? sender, PreviewKeyDownEventArgs e)
    {
        if (_renderingUnavailable)
        {
            return;
        }

        if (e.KeyCode is Keys.Left or Keys.Right or Keys.Up or Keys.Down or Keys.Tab)
        {
            e.IsInputKey = true;
        }
    }

    private void HandleTick(object? sender, EventArgs e)
    {
        if (_renderingUnavailable)
        {
            _deltaWatch.Restart();
            return;
        }

        var deltaTime = (float)_deltaWatch.Elapsed.TotalSeconds;
        _deltaWatch.Restart();

        _renderer.AdvanceTime(deltaTime);
        var updated = _flightMode ? UpdateFlight(deltaTime) : UpdateOrbit(deltaTime);
        var animated = _showObjects && _objectRenderer.Update(deltaTime);
        if (updated || animated)
        {
            _glControl.Invalidate();
        }
    }

    private bool UpdateOrbit(float deltaTime)
    {
        var updated = false;
        var azimuthSpeed = MathHelper.DegreesToRadians(90f) * deltaTime;
        var elevationSpeed = MathHelper.DegreesToRadians(60f) * deltaTime;
        var zoomSpeed = _camera.Distance * 1.5f * deltaTime;

        if (IsKeyDown(Keys.A) || IsKeyDown(Keys.Left))
        {
            _camera.Azimuth += azimuthSpeed;
            updated = true;
        }
        if (IsKeyDown(Keys.D) || IsKeyDown(Keys.Right))
        {
            _camera.Azimuth -= azimuthSpeed;
            updated = true;
        }
        if (IsKeyDown(Keys.W) || IsKeyDown(Keys.Up))
        {
            _camera.Elevation = Math.Clamp(_camera.Elevation + elevationSpeed, -1.2f, 1.2f);
            updated = true;
        }
        if (IsKeyDown(Keys.S) || IsKeyDown(Keys.Down))
        {
            _camera.Elevation = Math.Clamp(_camera.Elevation - elevationSpeed, -1.2f, 1.2f);
            updated = true;
        }

        if (IsKeyDown(Keys.Q) || IsKeyDown(Keys.PageUp))
        {
            _camera.Distance = Math.Clamp(_camera.Distance - zoomSpeed, 200f, 200000f);
            updated = true;
        }
        if (IsKeyDown(Keys.E) || IsKeyDown(Keys.PageDown))
        {
            _camera.Distance = Math.Clamp(_camera.Distance + zoomSpeed, 200f, 200000f);
            updated = true;
        }

        return updated;
    }

    private bool UpdateFlight(float deltaTime)
    {
        var forward = GetFlightForward();
        var right = Vector3.Cross(forward, Vector3.UnitZ);
        if (right.LengthSquared <= float.Epsilon)
        {
            right = Vector3.UnitX;
        }
        else
        {
            right = Vector3.Normalize(right);
        }
        var up = Vector3.Cross(right, forward);
        if (up.LengthSquared <= float.Epsilon)
        {
            up = Vector3.UnitZ;
        }
        else
        {
            up = Vector3.Normalize(up);
        }
        var move = Vector3.Zero;

        if (IsKeyDown(Keys.W))
        {
            move += forward;
        }
        if (IsKeyDown(Keys.S))
        {
            move -= forward;
        }
        if (IsKeyDown(Keys.A))
        {
            move -= right;
        }
        if (IsKeyDown(Keys.D))
        {
            move += right;
        }
        if (IsKeyDown(Keys.E) || IsKeyDown(Keys.Space))
        {
            move += up;
        }
        if (IsKeyDown(Keys.Q))
        {
            move -= up;
        }

        if (move != Vector3.Zero)
        {
            move = Vector3.Normalize(move);
        }

        var speed = _flightSpeed;
        if (IsKeyDown(Keys.ShiftKey))
        {
            speed *= 3f;
        }
        if (IsKeyDown(Keys.ControlKey))
        {
            speed *= 0.35f;
        }

        var updated = false;
        if (move != Vector3.Zero && deltaTime > 0f)
        {
            _flightPosition += move * speed * deltaTime;
            updated = true;
        }

        if (IsKeyDown(Keys.Z))
        {
            _flightYaw -= MathHelper.DegreesToRadians(90f) * deltaTime;
            updated = true;
        }
        if (IsKeyDown(Keys.C))
        {
            _flightYaw += MathHelper.DegreesToRadians(90f) * deltaTime;
            updated = true;
        }

        return updated;
    }

    private Matrix4 GetFlightViewMatrix()
    {
        var forward = GetFlightForward();
        var target = _flightPosition + forward;
        return Matrix4.LookAt(_flightPosition, target, Vector3.UnitZ);
    }

    private Vector3 GetFlightForward()
    {
        var cosPitch = MathF.Cos(_flightPitch);
        var sinPitch = MathF.Sin(_flightPitch);
        var cosYaw = MathF.Cos(_flightYaw);
        var sinYaw = MathF.Sin(_flightYaw);
        return Vector3.Normalize(new Vector3(cosPitch * cosYaw, cosPitch * sinYaw, sinPitch));
    }

    private bool IsKeyDown(Keys key) => _keys.Contains(key);

    private void SetFlightMode(bool enabled)
    {
        if (_flightMode == enabled)
        {
            return;
        }

        if (enabled)
        {
            SyncFlightFromOrbit();
            _flightMode = true;
            _glControl.Cursor = Cursors.Cross;
        }
        else
        {
            var forward = GetFlightForward();
            _camera.Azimuth = MathF.Atan2(-forward.Y, -forward.X);
            _camera.Elevation = Math.Clamp(MathF.Asin(-forward.Z), -1.2f, 1.2f);
            var distance = Math.Clamp(_camera.Distance, 200f, 200000f);
            _camera.Distance = distance;
            _camera.Target = _flightPosition + forward * distance;
            _flightMode = false;
            _glControl.Cursor = Cursors.Default;
        }

        _rotating = false;
        _panning = false;
        UpdateGameModeToggle();
    }

    private void SyncFlightFromOrbit()
    {
        _flightPosition = _camera.Position;
        var forward = _camera.Forward;
        _flightYaw = MathF.Atan2(forward.Y, forward.X);
        _flightPitch = MathF.Asin(forward.Z);
        _flightSpeed = Math.Clamp(_camera.Distance, 200f, 200000f);
    }

    private void UpdateGameModeToggle()
    {
        if (_gameModeToggle is null)
        {
            return;
        }

        _suppressGameModeEvent = true;
        _gameModeToggle.Checked = _flightMode;
        _suppressGameModeEvent = false;
    }

    private void GameModeToggleChanged(object? sender, EventArgs e)
    {
        if (_suppressGameModeEvent)
        {
            return;
        }

        SetFlightMode(_gameModeToggle.Checked);
    }

    private void ApplyRenderingSettings()
    {
        if (_renderingUnavailable)
        {
            return;
        }

        _renderer.SetFogEnabled(_fogEnabled);
        _renderer.SetLightingEnabled(_lightingEnabled);
        _objectRenderer.SetFogEnabled(_fogEnabled);
        _objectRenderer.SetLightingEnabled(_lightingEnabled);
        _objectRenderer.SetAnimationsEnabled(_animationsEnabled);
        _glControl.Invalidate();
    }

    private CheckBox CreateToggle(string text, EventHandler handler, bool isChecked)
    {
        var checkBox = new CheckBox
        {
            Text = text,
            AutoSize = true,
            Checked = isChecked,
            Margin = new Padding(0, 0, 12, 0),
        };
        checkBox.CheckedChanged += handler;
        return checkBox;
    }

    private static Vector3 EstimateFogColor(TextureImage? texture, MapContext context)
    {
        var baseColor = texture is null ? DefaultFogColor : Vector3.Lerp(DefaultFogColor, SampleAverageColor(texture), 0.6f);

        if (context.IsKarutan)
        {
            baseColor = Vector3.Lerp(baseColor, new Vector3(0.26f, 0.20f, 0.14f), 0.7f);
        }
        else if (context.IsBattleCastle)
        {
            baseColor = Vector3.Lerp(baseColor, new Vector3(0.18f, 0.22f, 0.30f), 0.5f);
        }
        else if (context.IsCryWolf)
        {
            baseColor = Vector3.Lerp(baseColor, new Vector3(0.22f, 0.27f, 0.36f), 0.4f);
        }
        else if (context.IsPkField)
        {
            baseColor = Vector3.Lerp(baseColor, new Vector3(0.32f, 0.24f, 0.19f), 0.5f);
        }

        return ClampColor(baseColor, 0.05f, 0.95f);
    }

    private static Vector2 EstimateFogParams(MapContext context)
    {
        var density = DefaultFogParams.X;
        var gradient = DefaultFogParams.Y;

        if (context.IsBattleCastle || context.IsCryWolf || context.IsDoppelGanger2)
        {
            density = 0.00060f;
            gradient = 1.45f;
        }
        else if (context.IsKarutan)
        {
            density = 0.00035f;
            gradient = 1.25f;
        }
        else if (context.IsPkField)
        {
            density = 0.00028f;
            gradient = 1.15f;
        }
        else if (context.IsCursedTemple)
        {
            density = 0.00052f;
            gradient = 1.40f;
        }

        return new Vector2(density, gradient);
    }

    private static Vector3 SampleAverageColor(TextureImage texture)
    {
        var pixels = texture.Pixels;
        var width = Math.Max(1, texture.Width);
        var height = Math.Max(1, texture.Height);
        var stepX = Math.Max(1, width / 128);
        var stepY = Math.Max(1, height / 128);
        double r = 0;
        double g = 0;
        double b = 0;
        var samples = 0;

        for (var y = 0; y < height; y += stepY)
        {
            for (var x = 0; x < width; x += stepX)
            {
                var offset = (y * width + x) * 4;
                if (offset + 2 >= pixels.Length)
                {
                    continue;
                }
                r += pixels[offset + 0];
                g += pixels[offset + 1];
                b += pixels[offset + 2];
                samples++;
            }
        }

        if (samples == 0)
        {
            return DefaultFogColor;
        }

        var inv = 1.0 / (samples * 255.0);
        return new Vector3((float)(r * inv), (float)(g * inv), (float)(b * inv));
    }

    private static Vector3 ClampColor(Vector3 color, float min, float max)
    {
        return new Vector3(
            MathHelper.Clamp(color.X, min, max),
            MathHelper.Clamp(color.Y, min, max),
            MathHelper.Clamp(color.Z, min, max));
    }

    private Control CreateFallbackSurface()
    {
        var panel = new Panel
        {
            Dock = DockStyle.Fill,
            BackColor = Color.Black,
        };
        panel.Paint += (_, e) => DrawFallbackMessage(e.Graphics);
        return panel;
    }

    private void ScheduleRenderingErrorMessage()
    {
        if (string.IsNullOrWhiteSpace(_renderingError) || _renderingErrorShown)
        {
            return;
        }

        void ShowMessage()
        {
            if (_renderingErrorShown)
            {
                return;
            }

            _renderingErrorShown = true;
            var owner = FindForm();
            var message = _renderingError + Environment.NewLine + Environment.NewLine +
                          "A visualização 3D foi desativada para evitar o fechamento do programa.";
            MessageBox.Show(owner, message, "Visualização 3D indisponível", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        if (!IsHandleCreated)
        {
            void Handler(object? sender, EventArgs args)
            {
                HandleCreated -= Handler;
                ShowMessage();
            }

            HandleCreated += Handler;
            return;
        }

        if (InvokeRequired)
        {
            BeginInvoke(new Action(ShowMessage));
        }
        else
        {
            ShowMessage();
        }
    }

    private void HandleRenderingFailure(string stage, Exception ex)
    {
        if (_renderingUnavailable)
        {
            return;
        }

        _renderingUnavailable = true;
        _contextReady = false;
        _renderingError = $"Não foi possível {stage}: {ex.Message}";
        _inputTimer.Stop();

        ScheduleRenderingErrorMessage();

        _glControl?.Invalidate();
    }

    private void DrawFallbackMessage(Graphics graphics)
    {
        graphics.Clear(Color.Black);
        if (string.IsNullOrWhiteSpace(_renderingError))
        {
            return;
        }

        using var brush = new SolidBrush(Color.White);
        using var format = new StringFormat
        {
            Alignment = StringAlignment.Center,
            LineAlignment = StringAlignment.Center,
        };
        graphics.DrawString(_renderingError, Font ?? SystemFonts.DefaultFont, brush,
            new RectangleF(0, 0, Width, Height), format);
    }
}
