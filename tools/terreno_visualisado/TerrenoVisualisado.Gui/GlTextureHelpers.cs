using System;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL4;

namespace TerrenoVisualisado.Gui;

internal static class GlTextureHelpers
{
    private const float DesiredAnisotropy = 8f;
    private static float? _maxSupportedAnisotropy;

    public static void ApplySamplerParameters(TextureTarget target, bool generateMipmaps)
    {
        var minFilter = generateMipmaps ? TextureMinFilter.LinearMipmapLinear : TextureMinFilter.Linear;
        GL.TexParameter(target, TextureParameterName.TextureMinFilter, (int)minFilter);
        GL.TexParameter(target, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        GL.TexParameter(target, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
        GL.TexParameter(target, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
        TryApplyAnisotropy(target);
        if (generateMipmaps)
        {
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);
        }
    }

    private static void TryApplyAnisotropy(TextureTarget target)
    {
        var max = QueryMaxSupportedAnisotropy();
        if (max <= 0f)
        {
            return;
        }

        var level = MathF.Min(DesiredAnisotropy, max);
        GL.TexParameter(target, (TextureParameterName)ExtTextureFilterAnisotropic.TextureMaxAnisotropyExt, level);
    }

    private static float QueryMaxSupportedAnisotropy()
    {
        if (_maxSupportedAnisotropy.HasValue)
        {
            return _maxSupportedAnisotropy.Value;
        }

        float max;
        try
        {
            max = GL.GetFloat((GetPName)ExtTextureFilterAnisotropic.MaxTextureMaxAnisotropyExt);
        }
        catch (GraphicsContextException)
        {
            max = 0f;
        }

        if (float.IsNaN(max) || max < 0f)
        {
            max = 0f;
        }

        _maxSupportedAnisotropy = max;
        return max;
    }
}
