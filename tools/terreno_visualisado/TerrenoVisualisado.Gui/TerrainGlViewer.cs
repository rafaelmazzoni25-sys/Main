using System;
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
    private TerrainMesh? _mesh;
    private bool _contextReady;
    private bool _rotating;
    private bool _panning;
    private Point _lastMouse;

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
        };

        Controls.Add(_glControl);

        _glControl.Load += HandleLoad;
        _glControl.Resize += HandleResize;
        _glControl.Paint += HandlePaint;
        _glControl.MouseDown += HandleMouseDown;
        _glControl.MouseUp += HandleMouseUp;
        _glControl.MouseMove += HandleMouseMove;
        _glControl.MouseWheel += HandleMouseWheel;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _renderer.Dispose();
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
            if (_contextReady)
            {
                _glControl.MakeCurrent();
                _renderer.EnsureResources();
                _glControl.Invalidate();
            }
            return;
        }

        _mesh = TerrainMeshBuilder.Build(world.Terrain);
        _renderer.UpdateData(_mesh, world.Visual?.CompositeTexture);
        ResetCamera();

        if (_contextReady)
        {
            _glControl.MakeCurrent();
            _renderer.EnsureResources();
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
    }

    private void HandleLoad(object? sender, EventArgs e)
    {
        _contextReady = true;
        _glControl.MakeCurrent();
        GL.ClearColor(0.1f, 0.14f, 0.2f, 1f);
        GL.Enable(EnableCap.DepthTest);
        _renderer.EnsureResources();
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
        var view = _camera.ViewMatrix;
        var projection = _camera.ProjectionMatrix(aspect);

        _renderer.EnsureResources();
        _renderer.Render(view, projection);

        _glControl.SwapBuffers();
    }

    private void HandleMouseDown(object? sender, MouseEventArgs e)
    {
        if (e.Button == MouseButtons.Left)
        {
            _rotating = true;
        }
        else if (e.Button == MouseButtons.Right)
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

        if (_rotating)
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
        _camera.Distance = Math.Clamp(_camera.Distance * factor, 200f, 200000f);
        _glControl.Invalidate();
    }
}
