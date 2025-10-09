using System;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using TerrenoVisualisado.Core;

namespace TerrenoVisualisado.Gui;

internal sealed class TerrainRenderer3D : IDisposable
{
    private TerrainMesh? _mesh;
    private TextureImage? _texture;
    private TextureImage? _lightMap;
    private bool _dirty = true;

    private int _vao;
    private int _vbo;
    private int _ebo;
    private int _textureHandle;
    private int _lightTextureHandle;
    private int _program;
    private int _uniformModel;
    private int _uniformView;
    private int _uniformProjection;
    private int _uniformLightDir;
    private int _uniformTexture;
    private int _uniformLightMap;
    private Vector3 _lightDirection = new(0.5f, -0.5f, 0.5f);
    private Vector3 _fogColor = new(0.23f, 0.28f, 0.34f);
    private Vector2 _fogParams = new(0.00045f, 1.35f);
    private bool _fogEnabled = true;
    private bool _lightingEnabled = true;
    private int _uniformCameraPosition;
    private int _uniformFogColor;
    private int _uniformFogParams;
    private int _uniformFogEnabled;
    private int _uniformLightingEnabled;
    private int _uniformTime;
    private float _elapsedTime;
    private int _indexCount;

    public void UpdateData(
        TerrainMesh? mesh,
        TextureImage? texture,
        TextureImage? lightMap,
        Vector3 lightDirection,
        Vector3 fogColor,
        Vector2 fogParams,
        bool fogEnabled,
        bool lightingEnabled)
    {
        _mesh = mesh;
        _texture = texture;
        _lightMap = lightMap;
        _lightDirection = lightDirection;
        _fogColor = fogColor;
        _fogParams = fogParams;
        _fogEnabled = fogEnabled;
        _lightingEnabled = lightingEnabled;
        _elapsedTime = 0f;
        _dirty = true;
    }

    public void SetFogEnabled(bool enabled)
    {
        _fogEnabled = enabled;
    }

    public void SetLightingEnabled(bool enabled)
    {
        _lightingEnabled = enabled;
    }

    public void AdvanceTime(float deltaTime)
    {
        if (deltaTime <= 0f)
        {
            return;
        }

        _elapsedTime += deltaTime;
        if (_elapsedTime > 7200f)
        {
            _elapsedTime -= 7200f;
        }
    }

    public void EnsureResources()
    {
        if (_mesh is null)
        {
            ReleaseResources();
            return;
        }

        if (!_dirty)
        {
            return;
        }

        ReleaseResources();

        _program = ShaderFactory.CreateProgram(VertexShaderSource, FragmentShaderSource);
        _uniformModel = GL.GetUniformLocation(_program, "uModel");
        _uniformView = GL.GetUniformLocation(_program, "uView");
        _uniformProjection = GL.GetUniformLocation(_program, "uProjection");
        _uniformLightDir = GL.GetUniformLocation(_program, "uLightDirection");
        _uniformTexture = GL.GetUniformLocation(_program, "uTexture");
        _uniformLightMap = GL.GetUniformLocation(_program, "uLightMap");
        _uniformCameraPosition = GL.GetUniformLocation(_program, "uCameraPosition");
        _uniformFogColor = GL.GetUniformLocation(_program, "uFogColor");
        _uniformFogParams = GL.GetUniformLocation(_program, "uFogParams");
        _uniformTime = GL.GetUniformLocation(_program, "uTime");
        _uniformFogEnabled = GL.GetUniformLocation(_program, "uFogEnabled");
        _uniformLightingEnabled = GL.GetUniformLocation(_program, "uLightingEnabled");

        _vao = GL.GenVertexArray();
        _vbo = GL.GenBuffer();
        _ebo = GL.GenBuffer();

        GL.BindVertexArray(_vao);

        GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);
        GL.BufferData(BufferTarget.ArrayBuffer, _mesh.Vertices.Length * sizeof(float), _mesh.Vertices, BufferUsageHint.StaticDraw);

        GL.BindBuffer(BufferTarget.ElementArrayBuffer, _ebo);
        GL.BufferData(BufferTarget.ElementArrayBuffer, _mesh.Indices.Length * sizeof(uint), _mesh.Indices, BufferUsageHint.StaticDraw);

        var stride = _mesh.VertexStride * sizeof(float);
        GL.EnableVertexAttribArray(0);
        GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, 0);
        GL.EnableVertexAttribArray(1);
        GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, 3 * sizeof(float));
        GL.EnableVertexAttribArray(2);
        GL.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, stride, 6 * sizeof(float));
        if (_mesh.VertexStride > 8)
        {
            GL.EnableVertexAttribArray(3);
            GL.VertexAttribPointer(3, 1, VertexAttribPointerType.Float, false, stride, 8 * sizeof(float));
        }
        else
        {
            GL.DisableVertexAttribArray(3);
        }

        var image = _texture ?? TextureImage.FromRgba(1, 1, new byte[] { 200, 200, 200, 255 });
        GL.ActiveTexture(TextureUnit.Texture0);
        _textureHandle = GL.GenTexture();
        GL.BindTexture(TextureTarget.Texture2D, _textureHandle);
        GL.PixelStore(PixelStoreParameter.UnpackAlignment, 1);
        GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, image.Width, image.Height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, image.Pixels);
        GlTextureHelpers.ApplySamplerParameters(TextureTarget.Texture2D, generateMipmaps: true);

        var lightImage = _lightMap ?? TextureImage.FromRgba(1, 1, new byte[] { 255, 255, 255, 255 });
        GL.ActiveTexture(TextureUnit.Texture1);
        _lightTextureHandle = GL.GenTexture();
        GL.BindTexture(TextureTarget.Texture2D, _lightTextureHandle);
        GL.PixelStore(PixelStoreParameter.UnpackAlignment, 1);
        GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, lightImage.Width, lightImage.Height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, lightImage.Pixels);
        GlTextureHelpers.ApplySamplerParameters(TextureTarget.Texture2D, generateMipmaps: false);

        _indexCount = _mesh.Indices.Length;
        _dirty = false;
    }

    public void Render(Matrix4 view, Matrix4 projection, Vector3 cameraPosition)
    {
        if (_mesh is null || _dirty)
        {
            return;
        }

        GL.Enable(EnableCap.DepthTest);
        GL.UseProgram(_program);

        var model = Matrix4.Identity;
        var viewMatrix = view;
        var projectionMatrix = projection;
        GL.UniformMatrix4(_uniformModel, false, ref model);
        GL.UniformMatrix4(_uniformView, false, ref viewMatrix);
        GL.UniformMatrix4(_uniformProjection, false, ref projectionMatrix);

        GL.Uniform3(_uniformLightDir, _lightDirection);
        GL.Uniform3(_uniformCameraPosition, cameraPosition);
        GL.Uniform3(_uniformFogColor, _fogColor);
        GL.Uniform2(_uniformFogParams, _fogParams);
        GL.Uniform1(_uniformTime, _elapsedTime);
        if (_uniformFogEnabled >= 0)
        {
            GL.Uniform1(_uniformFogEnabled, _fogEnabled ? 1 : 0);
        }
        if (_uniformLightingEnabled >= 0)
        {
            GL.Uniform1(_uniformLightingEnabled, _lightingEnabled ? 1 : 0);
        }

        GL.ActiveTexture(TextureUnit.Texture0);
        GL.BindTexture(TextureTarget.Texture2D, _textureHandle);
        GL.Uniform1(_uniformTexture, 0);

        GL.ActiveTexture(TextureUnit.Texture1);
        GL.BindTexture(TextureTarget.Texture2D, _lightTextureHandle);
        GL.Uniform1(_uniformLightMap, 1);

        GL.BindVertexArray(_vao);
        GL.DrawElements(PrimitiveType.Triangles, _indexCount, DrawElementsType.UnsignedInt, 0);
    }

    public void Dispose()
    {
        ReleaseResources();
        GC.SuppressFinalize(this);
    }

    private void ReleaseResources()
    {
        if (_program != 0)
        {
            GL.DeleteProgram(_program);
            _program = 0;
        }
        if (_vao != 0)
        {
            GL.DeleteVertexArray(_vao);
            _vao = 0;
        }
        if (_vbo != 0)
        {
            GL.DeleteBuffer(_vbo);
            _vbo = 0;
        }
        if (_ebo != 0)
        {
            GL.DeleteBuffer(_ebo);
            _ebo = 0;
        }
        if (_textureHandle != 0)
        {
            GL.DeleteTexture(_textureHandle);
            _textureHandle = 0;
        }
        if (_lightTextureHandle != 0)
        {
            GL.DeleteTexture(_lightTextureHandle);
            _lightTextureHandle = 0;
        }
        _indexCount = 0;
    }

    private const string VertexShaderSource = @"#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in float aMaterialFlags;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vNormal;
out vec2 vTexCoord;
out vec3 vWorldPos;
flat out int vMaterialFlags;

void main()
{
    vec4 worldPosition = uModel * vec4(aPosition, 1.0);
    vNormal = mat3(uModel) * aNormal;
    vTexCoord = aTexCoord;
    vWorldPos = worldPosition.xyz;
    vMaterialFlags = int(aMaterialFlags + 0.5);
    gl_Position = uProjection * uView * worldPosition;
}";

    private const string FragmentShaderSource = @"#version 330 core
in vec3 vNormal;
in vec2 vTexCoord;
in vec3 vWorldPos;
flat in int vMaterialFlags;

uniform sampler2D uTexture;
uniform sampler2D uLightMap;
uniform vec3 uLightDirection;
uniform vec3 uCameraPosition;
uniform vec3 uFogColor;
uniform vec2 uFogParams;
uniform float uTime;
uniform int uFogEnabled;
uniform int uLightingEnabled;

out vec4 FragColor;

void main()
{
    vec2 animatedUv = vTexCoord;
    if ((vMaterialFlags & 1) != 0)
    {
        animatedUv += vec2(uTime * 0.02, uTime * 0.015);
    }
    else if ((vMaterialFlags & 2) != 0)
    {
        animatedUv += vec2(uTime * 0.01, -uTime * 0.02);
    }

    vec4 baseColor = texture(uTexture, animatedUv);
    vec3 lightColor = texture(uLightMap, animatedUv).rgb;
    vec3 lit = baseColor.rgb * lightColor;
    if (uLightingEnabled != 0)
    {
        vec3 normal = normalize(vNormal);
        vec3 lightDir = normalize(-uLightDirection);
        vec3 viewDir = normalize(uCameraPosition - vWorldPos);
        vec3 halfDir = normalize(lightDir + viewDir);

        float diffuse = max(dot(normal, lightDir), 0.0);
        float ambient = 0.35;
        float emissive = ((vMaterialFlags & 16) != 0) ? 0.30 : 0.0;
        float specStrength = ((vMaterialFlags & 3) != 0) ? 0.55 : 0.20;
        float shininess = ((vMaterialFlags & 1) != 0) ? 48.0 : 24.0;
        float specular = pow(max(dot(normal, halfDir), 0.0), shininess) * specStrength;

        lit = baseColor.rgb * (ambient + diffuse * 0.75) * lightColor;
        lit += specular * lightColor;
        lit += emissive * baseColor.rgb;
    }

    float fogFactor = 1.0;
    if (uFogEnabled != 0)
    {
        float distanceToCamera = length(uCameraPosition - vWorldPos);
        fogFactor = exp(-pow(distanceToCamera * uFogParams.x, uFogParams.y));
        fogFactor = clamp(fogFactor, 0.0, 1.0);
    }
    vec3 finalColor = mix(uFogColor, lit, fogFactor);

    FragColor = vec4(finalColor, baseColor.a);
}";
}

internal static class ShaderFactory
{
    public static int CreateProgram(string vertexSource, string fragmentSource)
    {
        var vertex = CompileShader(ShaderType.VertexShader, vertexSource);
        var fragment = CompileShader(ShaderType.FragmentShader, fragmentSource);

        var program = GL.CreateProgram();
        GL.AttachShader(program, vertex);
        GL.AttachShader(program, fragment);
        GL.LinkProgram(program);

        GL.GetProgram(program, GetProgramParameterName.LinkStatus, out var status);
        if (status != (int)All.True)
        {
            var log = GL.GetProgramInfoLog(program);
            GL.DeleteShader(vertex);
            GL.DeleteShader(fragment);
            GL.DeleteProgram(program);
            throw new InvalidOperationException($"Falha ao linkar shader: {log}");
        }

        GL.DetachShader(program, vertex);
        GL.DetachShader(program, fragment);
        GL.DeleteShader(vertex);
        GL.DeleteShader(fragment);

        return program;
    }

    private static int CompileShader(ShaderType type, string source)
    {
        var shader = GL.CreateShader(type);
        GL.ShaderSource(shader, source);
        GL.CompileShader(shader);
        GL.GetShader(shader, ShaderParameter.CompileStatus, out var status);
        if (status != (int)All.True)
        {
            var log = GL.GetShaderInfoLog(shader);
            GL.DeleteShader(shader);
            throw new InvalidOperationException($"Falha ao compilar shader: {log}");
        }
        return shader;
    }
}
