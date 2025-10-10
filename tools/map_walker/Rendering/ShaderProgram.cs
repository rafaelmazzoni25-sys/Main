using System;
using OpenTK.Graphics.OpenGL4;

namespace MapWalker.Rendering;

internal sealed class ShaderProgram : IDisposable
{
    public ShaderProgram(string vertexSource, string fragmentSource)
    {
        Handle = GL.CreateProgram();
        var vertex = CompileShader(ShaderType.VertexShader, vertexSource);
        var fragment = CompileShader(ShaderType.FragmentShader, fragmentSource);
        GL.AttachShader(Handle, vertex);
        GL.AttachShader(Handle, fragment);
        GL.LinkProgram(Handle);
        GL.GetProgram(Handle, GetProgramParameterName.LinkStatus, out var status);
        GL.DetachShader(Handle, vertex);
        GL.DetachShader(Handle, fragment);
        GL.DeleteShader(vertex);
        GL.DeleteShader(fragment);
        if (status == 0)
        {
            var log = GL.GetProgramInfoLog(Handle);
            GL.DeleteProgram(Handle);
            throw new InvalidOperationException($"Falha ao linkar shader: {log}");
        }
    }

    public int Handle { get; }

    public void Use() => GL.UseProgram(Handle);

    public int GetUniformLocation(string name) => GL.GetUniformLocation(Handle, name);

    public void Dispose()
    {
        GL.DeleteProgram(Handle);
    }

    private static int CompileShader(ShaderType type, string source)
    {
        var handle = GL.CreateShader(type);
        GL.ShaderSource(handle, source);
        GL.CompileShader(handle);
        GL.GetShader(handle, ShaderParameter.CompileStatus, out var status);
        if (status == 0)
        {
            var log = GL.GetShaderInfoLog(handle);
            GL.DeleteShader(handle);
            throw new InvalidOperationException($"Falha ao compilar shader {type}: {log}");
        }

        return handle;
    }
}
