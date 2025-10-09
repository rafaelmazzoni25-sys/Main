using System;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;

namespace TerrenoVisualisado.Gui;

internal sealed class SkyRenderer : IDisposable
{
    private int _program;
    private int _vao;
    private int _uniformTopColor;
    private int _uniformBottomColor;
    private Vector3 _topColor = new(0.32f, 0.42f, 0.58f);
    private Vector3 _bottomColor = new(0.12f, 0.16f, 0.24f);

    public Vector3 BottomColor => _bottomColor;

    public void Configure(Vector3 fogColor)
    {
        var clamped = Vector3.Clamp(fogColor, Vector3.Zero, Vector3.One);
        _bottomColor = Vector3.Clamp(clamped * 0.9f + new Vector3(0.03f, 0.03f, 0.04f), Vector3.Zero, Vector3.One);
        _topColor = Vector3.Clamp(_bottomColor + new Vector3(0.18f, 0.18f, 0.22f), Vector3.Zero, Vector3.One);
    }

    public void EnsureResources()
    {
        if (_program != 0)
        {
            return;
        }

        _program = ShaderFactory.CreateProgram(VertexShaderSource, FragmentShaderSource);
        _uniformTopColor = GL.GetUniformLocation(_program, "uTopColor");
        _uniformBottomColor = GL.GetUniformLocation(_program, "uBottomColor");
        _vao = GL.GenVertexArray();
    }

    public void Render()
    {
        if (_program == 0)
        {
            return;
        }

        GL.DepthMask(false);
        GL.Disable(EnableCap.DepthTest);

        GL.UseProgram(_program);
        GL.Uniform3(_uniformTopColor, _topColor);
        GL.Uniform3(_uniformBottomColor, _bottomColor);
        GL.BindVertexArray(_vao);
        GL.DrawArrays(PrimitiveType.Triangles, 0, 3);
        GL.BindVertexArray(0);
        GL.UseProgram(0);

        GL.Enable(EnableCap.DepthTest);
        GL.DepthMask(true);
    }

    public void Dispose()
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
    }

    private const string VertexShaderSource = @"#version 330 core
const vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2(3.0, -1.0),
    vec2(-1.0, 3.0)
);

out vec2 vUv;

void main()
{
    vec2 pos = positions[gl_VertexID];
    gl_Position = vec4(pos, 0.0, 1.0);
    vUv = pos * 0.5 + 0.5;
}";

    private const string FragmentShaderSource = @"#version 330 core
in vec2 vUv;

uniform vec3 uTopColor;
uniform vec3 uBottomColor;

out vec4 FragColor;

void main()
{
    float t = clamp(vUv.y, 0.0, 1.0);
    vec3 color = mix(uBottomColor, uTopColor, pow(t, 0.82));
    FragColor = vec4(color, 1.0);
}";
}
