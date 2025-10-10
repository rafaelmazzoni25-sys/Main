using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using MapWalker.Terrain;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using OpenTK.WinForms;

namespace MapWalker.Rendering;

internal sealed class TerrainControl : GLControl
{
    private readonly Timer _timer;
    private readonly HashSet<Keys> _keys = new();
    private TerrainData? _terrain;
    private TerrainMesh? _mesh;
    private TileAtlas? _atlas;
    private ShaderProgram? _terrainProgram;
    private ShaderProgram? _gridProgram;
    private int _terrainVao;
    private int _terrainPositionVbo;
    private int _terrainNormalVbo;
    private int _terrainTexVbo;
    private int _terrainTileVbo;
    private int _terrainIndexBuffer;
    private int _terrainIndexCount;
    private int _gridVao;
    private int _gridVbo;
    private int _gridVertexCount;
    private int _objectsVao;
    private int _objectsVbo;
    private int _objectsVertexCount;
    private int _tileAtlasTexture;
    private int _tileLayer1Texture;
    private int _tileLayer2Texture;
    private int _tileAlphaTexture;
    private bool _tileAvailable;
    private int _tileCount;
    private Vector2 _atlasStep = Vector2.One;
    private Vector2 _atlasCounts = Vector2.One;
    private Vector3 _baseColor = Vector3.One;
    private float _eyeHeight = 120f;
    private float _moveSpeed = 800f;
    private bool _lockToGround = true;
    private bool _showGrid = true;
    private bool _showObjects = true;
    private bool _rotateActive;
    private Point _mousePosition;
    private Vector3 _cameraPosition = new(0f, 300f, 0f);
    private float _yaw = MathHelper.DegreesToRadians(-90f);
    private float _pitch = MathHelper.DegreesToRadians(-20f);
    private double _lastTick;
    private string? _renderError;

    public TerrainControl()
        : base(new GLControlSettings
        {
            MajorVersion = 3,
            MinorVersion = 3,
            Flags = ContextFlags.ForwardCompatible,
            Profile = ContextProfile.Core,
            NumberOfSamples = 4,
        })
    {
        BackColor = Color.Black;
        Dock = DockStyle.Fill;
        TabStop = true;
        _timer = new Timer { Interval = 16 };
        _timer.Tick += (_, _) => TickFrame();
        _timer.Start();
    }

    public event EventHandler<bool>? LockStateChanged;
    public event EventHandler<bool>? GridVisibilityChanged;
    public event EventHandler<bool>? ObjectVisibilityChanged;

    public float EyeHeight
    {
        get => _eyeHeight;
        set => _eyeHeight = value;
    }

    public float MoveSpeed
    {
        get => _moveSpeed;
        set => _moveSpeed = value;
    }

    public Vector3 BaseColor
    {
        get => _baseColor;
        set
        {
            _baseColor = value;
            Invalidate();
        }
    }

    public TerrainData? Terrain => _terrain;

    public Vector3 CameraPosition => _cameraPosition;

    public bool LockToGround
    {
        get => _lockToGround;
        set
        {
            if (_lockToGround == value)
            {
                return;
            }

            _lockToGround = value;
            LockStateChanged?.Invoke(this, value);
        }
    }

    public bool ShowGrid
    {
        get => _showGrid;
        set
        {
            if (_showGrid == value)
            {
                return;
            }

            _showGrid = value;
            GridVisibilityChanged?.Invoke(this, value);
        }
    }

    public bool ShowObjects
    {
        get => _showObjects;
        set
        {
            if (_showObjects == value)
            {
                return;
            }

            _showObjects = value;
            ObjectVisibilityChanged?.Invoke(this, value);
        }
    }

    public void SetTerrain(TerrainData? terrain)
    {
        MakeCurrent();
        ReleaseGlResources();
        _terrain = terrain;
        _mesh = terrain is null ? null : TerrainMeshBuilder.Build(terrain);
        _atlas = terrain is null || !terrain.HasTileMapping ? null : new TileAtlas(terrain.TileImages);
        if (terrain is not null)
        {
            UploadTerrain();
            UploadGrid();
            UploadObjects();
            UploadTiles();
            var extent = terrain.WorldExtent * 0.5f;
            var height = terrain.SampleHeight(extent, extent) + _eyeHeight;
            _cameraPosition = new Vector3(extent, height, extent);
        }

        Invalidate();
    }

    protected override void OnLoad(EventArgs e)
    {
        base.OnLoad(e);
        _lastTick = Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;
        try
        {
            CompilePrograms();
            GL.Enable(EnableCap.DepthTest);
            GL.Enable(EnableCap.CullFace);
            GL.CullFace(CullFaceMode.Back);
            GL.ClearColor(0.05f, 0.07f, 0.1f, 1f);
            if (_terrain is not null && _mesh is not null)
            {
                UploadTerrain();
                UploadGrid();
                UploadObjects();
                UploadTiles();
            }
        }
        catch (Exception ex)
        {
            _renderError = ex.Message;
        }
    }

    protected override void OnResize(EventArgs e)
    {
        base.OnResize(e);
        if (!IsHandleCreated)
        {
            return;
        }

        MakeCurrent();
        GL.Viewport(0, 0, Math.Max(1, Width), Math.Max(1, Height));
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        if (_renderError is not null)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            SwapBuffers();
            using var g = e.Graphics;
            using var brush = new SolidBrush(Color.White);
            using var format = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center };
            g.DrawString(_renderError, Font, brush, ClientRectangle, format);
            return;
        }

        if (_terrainProgram is null || _terrainVao == 0)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            SwapBuffers();
            return;
        }

        RenderScene();
        SwapBuffers();
    }

    protected override void OnKeyDown(KeyEventArgs e)
    {
        base.OnKeyDown(e);
        if (e.KeyCode == Keys.Space)
        {
            LockToGround = !LockToGround;
            return;
        }

        if (e.KeyCode == Keys.G)
        {
            ShowGrid = !ShowGrid;
            return;
        }

        if (e.KeyCode == Keys.O)
        {
            ShowObjects = !ShowObjects;
            return;
        }

        _keys.Add(e.KeyCode);
    }

    protected override void OnKeyUp(KeyEventArgs e)
    {
        base.OnKeyUp(e);
        _keys.Remove(e.KeyCode);
    }

    protected override void OnMouseDown(MouseEventArgs e)
    {
        base.OnMouseDown(e);
        if (e.Button == MouseButtons.Right)
        {
            Focus();
            _rotateActive = true;
            _mousePosition = e.Location;
            Cursor = Cursors.Cross;
        }
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (!_rotateActive)
        {
            return;
        }

        var delta = new Point(e.Location.X - _mousePosition.X, e.Location.Y - _mousePosition.Y);
        const float sensitivity = 0.005f;
        _yaw += delta.X * sensitivity;
        _pitch += delta.Y * sensitivity;
        _pitch = MathHelper.Clamp(_pitch, MathHelper.DegreesToRadians(-89f), MathHelper.DegreesToRadians(89f));
        _mousePosition = e.Location;
    }

    protected override void OnMouseUp(MouseEventArgs e)
    {
        base.OnMouseUp(e);
        if (e.Button == MouseButtons.Right)
        {
            _rotateActive = false;
            Cursor = Cursors.Default;
        }
    }

    protected override void OnMouseLeave(EventArgs e)
    {
        base.OnMouseLeave(e);
        if (_rotateActive)
        {
            _rotateActive = false;
            Cursor = Cursors.Default;
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _timer.Stop();
            _timer.Dispose();
        }

        ReleaseGlResources();
        _terrainProgram?.Dispose();
        _gridProgram?.Dispose();
        base.Dispose(disposing);
    }

    private void CompilePrograms()
    {
        _terrainProgram?.Dispose();
        _gridProgram?.Dispose();
        _terrainProgram = new ShaderProgram(VertexShaderSource, FragmentShaderSource);
        _gridProgram = new ShaderProgram(GridVertexSource, GridFragmentSource);
    }

    private void RenderScene()
    {
        GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
        if (_terrainProgram is null || _mesh is null)
        {
            return;
        }

        var projection = Matrix4.CreatePerspectiveFieldOfView(MathHelper.DegreesToRadians(60f), Math.Max(1f, Width) / Math.Max(1f, Height), 10f, 60000f);
        var view = CreateViewMatrix();
        var mvp = view * projection;
        var normalMatrix = Matrix3.Identity;

        _terrainProgram.Use();
        GL.BindVertexArray(_terrainVao);
        GL.UniformMatrix4(_terrainProgram.GetUniformLocation("u_mvp"), false, ref mvp);
        GL.UniformMatrix3(_terrainProgram.GetUniformLocation("u_normal"), false, ref normalMatrix);
        GL.Uniform3(_terrainProgram.GetUniformLocation("u_base_color"), _baseColor);
        var lightDir = new Vector3(-0.4f, 0.8f, -0.3f);
        lightDir.Normalize();
        GL.Uniform3(_terrainProgram.GetUniformLocation("u_light_direction"), lightDir);
        GL.Uniform1(_terrainProgram.GetUniformLocation("u_ambient"), 0.25f);
        GL.Uniform1(_terrainProgram.GetUniformLocation("u_tile_count"), Math.Max(_tileCount, 1));
        GL.Uniform1(_terrainProgram.GetUniformLocation("u_terrain_size"), _terrain?.Heights.GetLength(0) ?? TerrainData.TerrainSize);
        GL.Uniform2(_terrainProgram.GetUniformLocation("u_atlas_step"), _atlasStep);
        GL.Uniform2(_terrainProgram.GetUniformLocation("u_atlas_counts"), _atlasCounts);
        GL.Uniform1(_terrainProgram.GetUniformLocation("u_has_mapping"), _tileAvailable ? 1 : 0);
        if (_tileAvailable)
        {
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, _tileAtlasTexture);
            GL.Uniform1(_terrainProgram.GetUniformLocation("u_tile_atlas"), 0);
            GL.ActiveTexture(TextureUnit.Texture1);
            GL.BindTexture(TextureTarget.Texture2D, _tileLayer1Texture);
            GL.Uniform1(_terrainProgram.GetUniformLocation("u_tile_layer1"), 1);
            GL.ActiveTexture(TextureUnit.Texture2);
            GL.BindTexture(TextureTarget.Texture2D, _tileLayer2Texture);
            GL.Uniform1(_terrainProgram.GetUniformLocation("u_tile_layer2"), 2);
            GL.ActiveTexture(TextureUnit.Texture3);
            GL.BindTexture(TextureTarget.Texture2D, _tileAlphaTexture);
            GL.Uniform1(_terrainProgram.GetUniformLocation("u_tile_alpha"), 3);
        }

        GL.DrawElements(PrimitiveType.Triangles, _terrainIndexCount, DrawElementsType.UnsignedInt, 0);
        GL.BindVertexArray(0);
        GL.UseProgram(0);
        GL.BindTexture(TextureTarget.Texture2D, 0);
        GL.ActiveTexture(TextureUnit.Texture0);

        if (_showGrid && _gridProgram is not null && _gridVao != 0)
        {
            _gridProgram.Use();
            GL.UniformMatrix4(_gridProgram.GetUniformLocation("u_mvp"), false, ref mvp);
            var color = new Vector3(0.15f, 0.4f, 0.6f);
            GL.Uniform3(_gridProgram.GetUniformLocation("u_color"), color);
            GL.BindVertexArray(_gridVao);
            GL.DrawArrays(PrimitiveType.Lines, 0, _gridVertexCount);
            GL.BindVertexArray(0);
            GL.UseProgram(0);
        }

        if (_showObjects && _gridProgram is not null && _objectsVao != 0)
        {
            _gridProgram.Use();
            GL.UniformMatrix4(_gridProgram.GetUniformLocation("u_mvp"), false, ref mvp);
            var color = new Vector3(0.9f, 0.3f, 0.2f);
            GL.Uniform3(_gridProgram.GetUniformLocation("u_color"), color);
            GL.BindVertexArray(_objectsVao);
            GL.DrawArrays(PrimitiveType.Lines, 0, _objectsVertexCount);
            GL.BindVertexArray(0);
            GL.UseProgram(0);
        }
    }

    private Matrix4 CreateViewMatrix()
    {
        var forward = new Vector3(MathF.Cos(_yaw) * MathF.Cos(_pitch), MathF.Sin(-_pitch), MathF.Sin(_yaw) * MathF.Cos(_pitch));
        var target = _cameraPosition + forward;
        return Matrix4.LookAt(_cameraPosition, target, Vector3.UnitY);
    }

    private void TickFrame()
    {
        if (!IsHandleCreated || _terrain is null || _renderError is not null)
        {
            return;
        }

        var now = Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;
        var dt = (float)(now - _lastTick);
        _lastTick = now;
        if (dt <= 0)
        {
            return;
        }

        var speed = _moveSpeed;
        if (_keys.Contains(Keys.ShiftKey) || _keys.Contains(Keys.LShiftKey) || _keys.Contains(Keys.RShiftKey))
        {
            speed *= 1.8f;
        }

        var move = Vector3.Zero;
        var forward = new Vector3(MathF.Cos(_yaw), 0f, MathF.Sin(_yaw));
        var right = new Vector3(-MathF.Sin(_yaw), 0f, MathF.Cos(_yaw));
        if (_keys.Contains(Keys.W))
        {
            move += forward;
        }

        if (_keys.Contains(Keys.S))
        {
            move -= forward;
        }

        if (_keys.Contains(Keys.A))
        {
            move -= right;
        }

        if (_keys.Contains(Keys.D))
        {
            move += right;
        }

        if (move.LengthSquared > 0)
        {
            move.Normalize();
        }

        var displacement = move * speed * dt;
        var newPos = _cameraPosition + displacement;
        var extent = _terrain.WorldExtent;
        newPos.X = Math.Clamp(newPos.X, 0f, extent);
        newPos.Z = Math.Clamp(newPos.Z, 0f, extent);
        if (_terrain.IsWalkable(newPos.X, newPos.Z))
        {
            if (_lockToGround)
            {
                var ground = _terrain.SampleHeight(newPos.X, newPos.Z);
                newPos.Y = ground + _eyeHeight;
            }
            else
            {
                if (_keys.Contains(Keys.Q))
                {
                    newPos.Y -= speed * 0.5f * dt;
                }

                if (_keys.Contains(Keys.E))
                {
                    newPos.Y += speed * 0.5f * dt;
                }
            }

            _cameraPosition = newPos;
        }

        Invalidate();
    }

    private void UploadTerrain()
    {
        if (_mesh is null || _terrainProgram is null)
        {
            return;
        }

        _terrainVao = GL.GenVertexArray();
        GL.BindVertexArray(_terrainVao);

        _terrainPositionVbo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ArrayBuffer, _terrainPositionVbo);
        GL.BufferData(BufferTarget.ArrayBuffer, _mesh.Positions.Length * sizeof(float), _mesh.Positions, BufferUsageHint.StaticDraw);
        GL.EnableVertexAttribArray(0);
        GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 0, 0);

        _terrainNormalVbo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ArrayBuffer, _terrainNormalVbo);
        GL.BufferData(BufferTarget.ArrayBuffer, _mesh.Normals.Length * sizeof(float), _mesh.Normals, BufferUsageHint.StaticDraw);
        GL.EnableVertexAttribArray(1);
        GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 0, 0);

        _terrainTexVbo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ArrayBuffer, _terrainTexVbo);
        GL.BufferData(BufferTarget.ArrayBuffer, _mesh.TexCoords.Length * sizeof(float), _mesh.TexCoords, BufferUsageHint.StaticDraw);
        GL.EnableVertexAttribArray(2);
        GL.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, 0, 0);

        _terrainTileVbo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ArrayBuffer, _terrainTileVbo);
        GL.BufferData(BufferTarget.ArrayBuffer, _mesh.TileCoords.Length * sizeof(float), _mesh.TileCoords, BufferUsageHint.StaticDraw);
        GL.EnableVertexAttribArray(3);
        GL.VertexAttribPointer(3, 2, VertexAttribPointerType.Float, false, 0, 0);

        _terrainIndexBuffer = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ElementArrayBuffer, _terrainIndexBuffer);
        GL.BufferData(BufferTarget.ElementArrayBuffer, _mesh.Indices.Length * sizeof(uint), _mesh.Indices, BufferUsageHint.StaticDraw);
        _terrainIndexCount = _mesh.Indices.Length;

        GL.BindVertexArray(0);
        GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
    }

    private void UploadGrid()
    {
        if (_terrain is null)
        {
            return;
        }

        var extent = _terrain.WorldExtent;
        var size = _terrain.Size;
        var step = Math.Max(1, size / 16);
        var baseHeight = _terrain.Heights.Cast<float>().DefaultIfEmpty().Min() - 5f;
        var vertices = new List<Vector3>();
        for (var i = 0; i < size; i += step)
        {
            var x = i * TerrainData.TerrainScale;
            vertices.Add(new Vector3(x, baseHeight, 0f));
            vertices.Add(new Vector3(x, baseHeight, extent));
            vertices.Add(new Vector3(0f, baseHeight, x));
            vertices.Add(new Vector3(extent, baseHeight, x));
        }

        if (vertices.Count == 0)
        {
            _gridVao = 0;
            _gridVbo = 0;
            _gridVertexCount = 0;
            return;
        }

        var buffer = new float[vertices.Count * 3];
        for (var i = 0; i < vertices.Count; i++)
        {
            buffer[i * 3 + 0] = vertices[i].X;
            buffer[i * 3 + 1] = vertices[i].Y;
            buffer[i * 3 + 2] = vertices[i].Z;
        }

        _gridVao = GL.GenVertexArray();
        _gridVbo = GL.GenBuffer();
        _gridVertexCount = vertices.Count;
        GL.BindVertexArray(_gridVao);
        GL.BindBuffer(BufferTarget.ArrayBuffer, _gridVbo);
        GL.BufferData(BufferTarget.ArrayBuffer, buffer.Length * sizeof(float), buffer, BufferUsageHint.StaticDraw);
        GL.EnableVertexAttribArray(0);
        GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 0, 0);
        GL.BindVertexArray(0);
    }

    private void UploadObjects()
    {
        if (_terrain is null || _terrain.Objects.Count == 0)
        {
            _objectsVao = 0;
            _objectsVbo = 0;
            _objectsVertexCount = 0;
            return;
        }

        var vertices = new List<Vector3>();
        foreach (var obj in _terrain.Objects)
        {
            vertices.Add(obj.Position);
            vertices.Add(obj.Position + new Vector3(0f, 150f, 0f));
        }

        var buffer = new float[vertices.Count * 3];
        for (var i = 0; i < vertices.Count; i++)
        {
            buffer[i * 3 + 0] = vertices[i].X;
            buffer[i * 3 + 1] = vertices[i].Y;
            buffer[i * 3 + 2] = vertices[i].Z;
        }

        _objectsVao = GL.GenVertexArray();
        _objectsVbo = GL.GenBuffer();
        _objectsVertexCount = vertices.Count;
        GL.BindVertexArray(_objectsVao);
        GL.BindBuffer(BufferTarget.ArrayBuffer, _objectsVbo);
        GL.BufferData(BufferTarget.ArrayBuffer, buffer.Length * sizeof(float), buffer, BufferUsageHint.StaticDraw);
        GL.EnableVertexAttribArray(0);
        GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 0, 0);
        GL.BindVertexArray(0);
    }

    private void UploadTiles()
    {
        DeleteTileTextures();
        _tileAvailable = _terrain is not null && _terrain.HasTileMapping;
        if (!_tileAvailable || _terrain is null || _atlas is null)
        {
            return;
        }

        _tileCount = _terrain.TileImages.Count;
        _atlasStep = new Vector2(1f / _atlas.Columns, 1f / _atlas.Rows);
        _atlasCounts = new Vector2(_atlas.Columns, _atlas.Rows);

        GL.PixelStore(PixelStoreParameter.UnpackAlignment, 1);
        _tileAtlasTexture = GL.GenTexture();
        GL.BindTexture(TextureTarget.Texture2D, _tileAtlasTexture);
        GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, _atlas.Width, _atlas.Height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, _atlas.Pixels);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

        _tileLayer1Texture = CreateIndexTexture(_terrain.TileLayer1!);
        _tileLayer2Texture = CreateIndexTexture(_terrain.TileLayer2!);
        _tileAlphaTexture = CreateAlphaTexture(_terrain.TileAlpha!);
        GL.BindTexture(TextureTarget.Texture2D, 0);
    }

    private static int CreateIndexTexture(byte[,] data)
    {
        var height = data.GetLength(0);
        var width = data.GetLength(1);
        var flat = new byte[width * height];
        var idx = 0;
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                flat[idx++] = data[y, x];
            }
        }

        var texture = GL.GenTexture();
        GL.BindTexture(TextureTarget.Texture2D, texture);
        GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.R8, width, height, 0, PixelFormat.Red, PixelType.UnsignedByte, flat);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        return texture;
    }

    private static int CreateAlphaTexture(float[,] data)
    {
        var height = data.GetLength(0);
        var width = data.GetLength(1);
        var flat = new byte[width * height];
        var idx = 0;
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var value = Math.Clamp(data[y, x], 0f, 1f);
                flat[idx++] = (byte)Math.Round(value * 255f);
            }
        }

        var texture = GL.GenTexture();
        GL.BindTexture(TextureTarget.Texture2D, texture);
        GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.R8, width, height, 0, PixelFormat.Red, PixelType.UnsignedByte, flat);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        return texture;
    }

    private void DeleteTileTextures()
    {
        if (_tileAtlasTexture != 0)
        {
            GL.DeleteTexture(_tileAtlasTexture);
            _tileAtlasTexture = 0;
        }

        if (_tileLayer1Texture != 0)
        {
            GL.DeleteTexture(_tileLayer1Texture);
            _tileLayer1Texture = 0;
        }

        if (_tileLayer2Texture != 0)
        {
            GL.DeleteTexture(_tileLayer2Texture);
            _tileLayer2Texture = 0;
        }

        if (_tileAlphaTexture != 0)
        {
            GL.DeleteTexture(_tileAlphaTexture);
            _tileAlphaTexture = 0;
        }

        _tileAvailable = false;
        _tileCount = 0;
    }

    private void ReleaseGlResources()
    {
        DeleteTileTextures();

        if (_terrainIndexBuffer != 0)
        {
            GL.DeleteBuffer(_terrainIndexBuffer);
            _terrainIndexBuffer = 0;
        }

        if (_terrainTileVbo != 0)
        {
            GL.DeleteBuffer(_terrainTileVbo);
            _terrainTileVbo = 0;
        }

        if (_terrainTexVbo != 0)
        {
            GL.DeleteBuffer(_terrainTexVbo);
            _terrainTexVbo = 0;
        }

        if (_terrainNormalVbo != 0)
        {
            GL.DeleteBuffer(_terrainNormalVbo);
            _terrainNormalVbo = 0;
        }

        if (_terrainPositionVbo != 0)
        {
            GL.DeleteBuffer(_terrainPositionVbo);
            _terrainPositionVbo = 0;
        }

        if (_terrainVao != 0)
        {
            GL.DeleteVertexArray(_terrainVao);
            _terrainVao = 0;
        }

        if (_gridVbo != 0)
        {
            GL.DeleteBuffer(_gridVbo);
            _gridVbo = 0;
        }

        if (_gridVao != 0)
        {
            GL.DeleteVertexArray(_gridVao);
            _gridVao = 0;
        }

        if (_objectsVbo != 0)
        {
            GL.DeleteBuffer(_objectsVbo);
            _objectsVbo = 0;
        }

        if (_objectsVao != 0)
        {
            GL.DeleteVertexArray(_objectsVao);
            _objectsVao = 0;
        }
    }

    private const string VertexShaderSource = @"#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texcoord;
layout(location = 3) in vec2 a_tilecoord;
uniform mat4 u_mvp;
uniform mat3 u_normal;
uniform vec3 u_light_direction;
out float v_light;
out vec2 v_texcoord;
out vec2 v_tilecoord;
void main() {
    vec3 n = normalize(u_normal * a_normal);
    vec3 lightDir = normalize(u_light_direction);
    v_light = max(dot(n, lightDir), 0.1);
    v_texcoord = a_texcoord;
    v_tilecoord = a_tilecoord;
    gl_Position = u_mvp * vec4(a_position, 1.0);
}";

    private const string FragmentShaderSource = @"#version 330 core
uniform vec3 u_base_color;
uniform float u_ambient;
uniform bool u_has_mapping;
uniform float u_tile_count;
uniform float u_terrain_size;
uniform vec2 u_atlas_step;
uniform vec2 u_atlas_counts;
uniform sampler2D u_tile_atlas;
uniform sampler2D u_tile_layer1;
uniform sampler2D u_tile_layer2;
uniform sampler2D u_tile_alpha;
in float v_light;
in vec2 v_texcoord;
in vec2 v_tilecoord;
out vec4 fragColor;
void main() {
    float diffuse = max(v_light, u_ambient);
    vec3 color = u_base_color;
    if (u_has_mapping) {
        vec2 cell = clamp(floor(v_tilecoord), vec2(0.0), vec2(u_terrain_size - 1.0));
        vec2 lookup = (cell + 0.5) / u_terrain_size;
        float idx1 = texture(u_tile_layer1, lookup).r * 255.0;
        float idx2 = texture(u_tile_layer2, lookup).r * 255.0;
        float alpha = texture(u_tile_alpha, lookup).r;
        float tileIdx1 = clamp(floor(idx1 + 0.5), 0.0, u_tile_count - 1.0);
        float tileIdx2 = floor(idx2 + 0.5);
        vec2 frac_uv = fract(v_texcoord);
        float cols = u_atlas_counts.x;
        float rows = u_atlas_counts.y;
        vec2 step = u_atlas_step;
        float col1 = mod(tileIdx1, cols);
        float row1 = floor(tileIdx1 / cols);
        vec2 base1 = vec2(col1, row1) * step;
        vec3 tex1 = texture(u_tile_atlas, base1 + frac_uv * step).rgb;
        vec3 texColor = tex1;
        if (alpha > 0.001 && tileIdx2 >= 0.0 && tileIdx2 < u_tile_count) {
            float col2 = mod(tileIdx2, cols);
            float row2 = floor(tileIdx2 / cols);
            vec2 base2 = vec2(col2, row2) * step;
            vec3 tex2 = texture(u_tile_atlas, base2 + frac_uv * step).rgb;
            texColor = mix(tex1, tex2, clamp(alpha, 0.0, 1.0));
        }
        color = texColor * u_base_color;
    }
    fragColor = vec4(color * diffuse, 1.0);
}";

    private const string GridVertexSource = @"#version 330 core
layout(location = 0) in vec3 a_position;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
}";

    private const string GridFragmentSource = @"#version 330 core
uniform vec3 u_color;
out vec4 fragColor;
void main() {
    fragColor = vec4(u_color, 1.0);
}";
}
