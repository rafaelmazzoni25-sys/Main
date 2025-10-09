using System;
using OpenTK.Mathematics;
using TerrenoVisualisado.Core;

namespace TerrenoVisualisado.Gui;

internal readonly struct LightingProfile
{
    private static readonly Vector3 s_defaultDirection = Vector3.Normalize(new Vector3(0.5f, 0.5f, -0.5f));
    private static readonly Vector3 s_defaultSunColor = new(1.05f, 0.98f, 0.92f);
    private static readonly Vector3 s_defaultAmbientColor = new(0.35f, 0.38f, 0.42f);
    private static readonly Vector3 s_defaultSpecularColor = new(0.90f, 0.90f, 0.88f);

    public static LightingProfile Default { get; } = new LightingProfile(
        s_defaultDirection,
        ClampColor(s_defaultSunColor),
        ClampColor(s_defaultAmbientColor),
        ClampColor(s_defaultSpecularColor),
        0.85f,
        0.55f,
        0.32f);

    public LightingProfile(
        Vector3 direction,
        Vector3 sunColor,
        Vector3 ambientColor,
        Vector3 specularColor,
        float diffuseStrength,
        float specularStrength,
        float emissiveStrength)
    {
        Direction = Normalize(direction);
        SunColor = ClampColor(sunColor);
        AmbientColor = ClampColor(ambientColor);
        SpecularColor = ClampColor(specularColor);
        DiffuseStrength = ClampIntensity(diffuseStrength);
        SpecularStrength = ClampIntensity(specularStrength);
        EmissiveStrength = ClampIntensity(emissiveStrength);
    }

    public Vector3 Direction { get; }
    public Vector3 SunColor { get; }
    public Vector3 AmbientColor { get; }
    public Vector3 SpecularColor { get; }
    public float DiffuseStrength { get; }
    public float SpecularStrength { get; }
    public float EmissiveStrength { get; }

    public static LightingProfile Create(Vector3 lightDirection, Vector3 fogColor, MapContext context)
    {
        var direction = Normalize(lightDirection);
        var fog = ClampColor(fogColor);

        var sunColor = ClampColor(Vector3.Lerp(Default.SunColor, fog * 0.6f + new Vector3(0.18f, 0.18f, 0.20f), 0.25f));
        var ambient = ClampColor(Vector3.Lerp(Default.AmbientColor, fog * 0.9f + new Vector3(0.04f), 0.6f));
        var specular = ClampColor(Vector3.Lerp(Default.SpecularColor, sunColor, 0.35f));
        var diffuseStrength = Default.DiffuseStrength;
        var specularStrength = Default.SpecularStrength;
        var emissiveStrength = Default.EmissiveStrength;

        if (context.IsKarutan)
        {
            sunColor = ClampColor(new Vector3(1.14f, 0.74f, 0.54f));
            ambient = ClampColor(Vector3.Lerp(ambient, new Vector3(0.42f, 0.29f, 0.18f), 0.55f));
            diffuseStrength = 0.78f;
            emissiveStrength = 0.36f;
        }
        else if (context.IsCryWolf)
        {
            sunColor = ClampColor(new Vector3(0.78f, 0.88f, 1.12f));
            ambient = ClampColor(Vector3.Lerp(ambient, new Vector3(0.28f, 0.34f, 0.46f), 0.6f));
            specular = ClampColor(Vector3.Lerp(specular, new Vector3(0.58f, 0.68f, 0.90f), 0.5f));
            diffuseStrength = 0.82f;
        }
        else if (context.IsBattleCastle)
        {
            sunColor = ClampColor(new Vector3(0.86f, 0.96f, 1.10f));
            diffuseStrength = 0.72f;
            specularStrength = 0.62f;
        }
        else if (context.IsPkField)
        {
            sunColor = ClampColor(new Vector3(1.12f, 0.92f, 0.78f));
            ambient = ClampColor(Vector3.Lerp(ambient, new Vector3(0.36f, 0.28f, 0.18f), 0.5f));
        }
        else if (context.IsEmpireGuardian)
        {
            diffuseStrength = 0.90f;
            specularStrength = 0.68f;
        }

        return new LightingProfile(direction, sunColor, ambient, specular, diffuseStrength, specularStrength, emissiveStrength);
    }

    private static Vector3 Normalize(Vector3 direction)
    {
        var lengthSquared = direction.LengthSquared;
        if (lengthSquared < 1e-4f)
        {
            return s_defaultDirection;
        }

        var invLength = 1.0f / MathF.Sqrt(lengthSquared);
        return direction * invLength;
    }

    private static Vector3 ClampColor(Vector3 color)
    {
        return new Vector3(
            MathHelper.Clamp(color.X, 0.0f, 1.25f),
            MathHelper.Clamp(color.Y, 0.0f, 1.25f),
            MathHelper.Clamp(color.Z, 0.0f, 1.25f));
    }

    private static float ClampIntensity(float value)
    {
        return Math.Clamp(value, 0.0f, 2.0f);
    }
}
