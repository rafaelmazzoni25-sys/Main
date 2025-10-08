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
    private readonly GLControl _glControl;
    private readonly OrbitCamera _camera = new();
    private readonly TerrainRenderer3D _renderer = new();
    private readonly ObjectRenderer3D _objectRenderer = new();
    private readonly Timer _inputTimer;
    private readonly Stopwatch _deltaWatch = Stopwatch.StartNew();
    private readonly HashSet<Keys> _keys = new();
    private TerrainMesh? _mesh;
    private bool _contextReady;
    private bool _rotating;
    private bool _panning;
    private Point _lastMouse;
    private bool _flightMode;
    private float _flightYaw;
    private float _flightPitch;
    private float _flightSpeed = 1000f;
    private Vector3 _flightPosition;

    public TerrainGlViewer()
    {
        Dock = DockStyle.Fill;
        DoubleBuffered = true;

        _glControl = new GLControl(new GLControlSettings
        {
            MajorVersion = 4,
            MinorVersion = 1,
            Flags = ContextFlags.Default,
        })
        {
            Dock = DockStyle.Fill,
            BackColor = Color.Black,
            TabStop = true,
        };

        Controls.Add(_glControl);

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

        _inputTimer = new Timer
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
            _renderer.Dispose();
            _objectRenderer.Dispose();
            _glControl.Dispose();
        }
        base.Dispose(disposing);
    }

    public void DisplayWorld(WorldData? world)
    {
        if (world is null)
        {
            _mesh = null;
            _renderer.UpdateData(null, null);
            _objectRenderer.UpdateWorld(null);
            _flightMode = false;
            _rotating = false;
            _panning = false;
            if (_contextReady)
            {
                _glControl.MakeCurrent();
                _renderer.EnsureResources();
                _objectRenderer.EnsureResources();
                _glControl.Invalidate();
            }
            return;
        }

        _mesh = TerrainMeshBuilder.Build(world.Terrain);
        _renderer.UpdateData(_mesh, world.Visual?.CompositeTexture);
        _objectRenderer.UpdateWorld(world);
        ResetCamera();

        if (_contextReady)
        {
            _glControl.MakeCurrent();
            _renderer.EnsureResources();
            _objectRenderer.EnsureResources();
            _glControl.Invalidate();
        }
    }

    private void ResetCamera()
    {
        if (_mesh is null)
        {
            return;
        }

        var extent = (WorldLoader.TerrainSize - 1) * WorldLoader.TerrainScale;
        var center = new Vector3(extent * 0.5f, (_mesh.BoundsMin.Y + _mesh.BoundsMax.Y) * 0.5f, extent * 0.5f);
        _camera.Target = center;
        _camera.Azimuth = MathHelper.DegreesToRadians(45f);
        _camera.Elevation = MathHelper.DegreesToRadians(35f);
        var radius = Math.Max(extent, _mesh.BoundsMax.Y - _mesh.BoundsMin.Y);
        _camera.Distance = Math.Max(1000f, radius * 1.2f);
        _flightMode = false;
        _rotating = false;
        _panning = false;
        SyncFlightFromOrbit();
    }

    private void HandleLoad(object? sender, EventArgs e)
    {
        _contextReady = true;
        _glControl.MakeCurrent();
        GL.ClearColor(0.1f, 0.14f, 0.2f, 1f);
        GL.Enable(EnableCap.DepthTest);
        _renderer.EnsureResources();
        _objectRenderer.EnsureResources();
    }

    private void HandleResize(object? sender, EventArgs e)
    {
        if (!_contextReady)
        {
            return;
        }

        _glControl.MakeCurrent();
        GL.Viewport(0, 0, Math.Max(1, _glControl.Width), Math.Max(1, _glControl.Height));
        _glControl.Invalidate();
    }

    private void HandlePaint(object? sender, PaintEventArgs e)
    {
        if (!_contextReady)
        {
            e.Graphics.Clear(Color.Black);
            return;
        }

        _glControl.MakeCurrent();
        GL.Viewport(0, 0, Math.Max(1, _glControl.Width), Math.Max(1, _glControl.Height));
        GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

        if (_mesh is null)
        {
            _glControl.SwapBuffers();
            return;
        }

        var aspect = _glControl.Width / (float)Math.Max(1, _glControl.Height);
        var view = _flightMode ? GetFlightViewMatrix() : _camera.ViewMatrix;
        var projection = _camera.ProjectionMatrix(aspect);

        _renderer.EnsureResources();
        _renderer.Render(view, projection);
        _objectRenderer.EnsureResources();
        _objectRenderer.Render(view, projection);

        _glControl.SwapBuffers();
    }

    private void HandleMouseDown(object? sender, MouseEventArgs e)
    {
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
        _keys.Add(e.KeyCode);
        if (e.KeyCode == Keys.F)
        {
            ToggleFlightMode();
            e.Handled = true;
        }
        else if (e.KeyCode == Keys.R)
        {
            ResetCamera();
            if (_flightMode)
            {
                SyncFlightFromOrbit();
            }
            e.Handled = true;
        }
    }

    private void HandleKeyUp(object? sender, KeyEventArgs e)
    {
        _keys.Remove(e.KeyCode);
    }

    private void HandleLostFocus(object? sender, EventArgs e)
    {
        _keys.Clear();
        _rotating = false;
        _panning = false;
    }

    private void HandlePreviewKeyDown(object? sender, PreviewKeyDownEventArgs e)
    {
        if (e.KeyCode is Keys.Left or Keys.Right or Keys.Up or Keys.Down or Keys.Tab)
        {
            e.IsInputKey = true;
        }
    }

    private void HandleTick(object? sender, EventArgs e)
    {
        var deltaTime = (float)_deltaWatch.Elapsed.TotalSeconds;
        _deltaWatch.Restart();

        var updated = _flightMode ? UpdateFlight(deltaTime) : UpdateOrbit(deltaTime);
        var animated = _objectRenderer.Update(deltaTime);
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
        var right = Vector3.Cross(forward, Vector3.UnitY);
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
            up = Vector3.UnitY;
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
        if (IsKeyDown(Keys.E))
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
        return Matrix4.LookAt(_flightPosition, target, Vector3.UnitY);
    }

    private Vector3 GetFlightForward()
    {
        var cosPitch = MathF.Cos(_flightPitch);
        var sinPitch = MathF.Sin(_flightPitch);
        var cosYaw = MathF.Cos(_flightYaw);
        var sinYaw = MathF.Sin(_flightYaw);
        return Vector3.Normalize(new Vector3(cosPitch * cosYaw, sinPitch, cosPitch * sinYaw));
    }

    private bool IsKeyDown(Keys key) => _keys.Contains(key);

    private void ToggleFlightMode()
    {
        if (_flightMode)
        {
            _flightMode = false;
            var forward = GetFlightForward();
            _camera.Azimuth = MathF.Atan2(-forward.Z, -forward.X);
            _camera.Elevation = Math.Clamp(MathF.Asin(-forward.Y), -1.2f, 1.2f);
            var distance = Math.Clamp(_camera.Distance, 200f, 200000f);
            _camera.Distance = distance;
            _camera.Target = _flightPosition + forward * distance;
        }
        else
        {
            SyncFlightFromOrbit();
            _flightMode = true;
        }

        _rotating = false;
        _panning = false;
    }

    private void SyncFlightFromOrbit()
    {
        _flightPosition = _camera.Position;
        var forward = _camera.Forward;
        _flightYaw = MathF.Atan2(forward.Z, forward.X);
        _flightPitch = MathF.Asin(forward.Y);
        _flightSpeed = Math.Clamp(_camera.Distance, 200f, 200000f);
    }
}
