using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using TerrenoVisualisado.Core;

namespace TerrenoVisualisado.Gui;

internal sealed class ObjectRenderer3D : IDisposable
{
    private const int MaxBones = 64;
    private const float AnimationFps = 30f;

    private readonly Dictionary<short, ModelResources> _modelResources = new();
    private readonly Dictionary<string, int> _textureCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly Matrix4[] _bonePalette = Enumerable.Repeat(Matrix4.Identity, MaxBones).ToArray();

    private WorldData? _world;
    private TextureLibrary? _textureLibrary;
    private bool _dirty = true;
    private bool _hasAnimatedModels;
    private float _animationTime;
    private Vector3 _lightDirection = new(-0.4f, -1.0f, -0.6f);
    private Vector3 _fogColor = new(0.23f, 0.28f, 0.34f);
    private Vector2 _fogParams = new(0.00045f, 1.35f);
    private bool _fogEnabled = true;
    private bool _lightingEnabled = true;
    private bool _animationsEnabled = true;

    private int _program;
    private int _uniformModel;
    private int _uniformView;
    private int _uniformProjection;
    private int _uniformBones;
    private int _uniformLightDirection;
    private int _uniformAlphaCutoff;
    private int _uniformMaterialFlags;
    private int _uniformTexture;
    private int _uniformCameraPosition;
    private int _uniformFogColor;
    private int _uniformFogParams;
    private int _uniformTime;
    private int _uniformFogEnabled;
    private int _uniformLightingEnabled;

    public void UpdateWorld(WorldData? world)
    {
        _world = world;
        _dirty = true;
    }

    public void ConfigureEnvironment(Vector3 lightDirection, Vector3 fogColor, Vector2 fogParams, bool fogEnabled, bool lightingEnabled)
    {
        _lightDirection = lightDirection;
        _fogColor = fogColor;
        _fogParams = fogParams;
        _fogEnabled = fogEnabled;
        _lightingEnabled = lightingEnabled;
    }

    public bool Update(float deltaTime)
    {
        if (_world is null)
        {
            return false;
        }

        if (!_animationsEnabled || deltaTime <= 0f)
        {
            return false;
        }

        _animationTime += deltaTime;
        if (_animationTime > 3600f)
        {
            _animationTime -= 3600f;
        }

        return _hasAnimatedModels;
    }

    public void SetFogEnabled(bool enabled)
    {
        _fogEnabled = enabled;
    }

    public void SetLightingEnabled(bool enabled)
    {
        _lightingEnabled = enabled;
    }

    public void SetAnimationsEnabled(bool enabled)
    {
        _animationsEnabled = enabled;
    }

    public void EnsureResources()
    {
        if (_dirty)
        {
            RebuildResources();
        }
    }

    public void Render(Matrix4 view, Matrix4 projection, Vector3 cameraPosition)
    {
        if (_world is null || _program == 0 || _modelResources.Count == 0)
        {
            return;
        }

        GL.UseProgram(_program);
        GL.UniformMatrix4(_uniformView, false, ref view);
        GL.UniformMatrix4(_uniformProjection, false, ref projection);
        GL.Uniform3(_uniformLightDirection, _lightDirection);
        GL.Uniform3(_uniformCameraPosition, cameraPosition);
        GL.Uniform3(_uniformFogColor, _fogColor);
        GL.Uniform2(_uniformFogParams, _fogParams);
        GL.Uniform1(_uniformTime, _animationTime);
        if (_uniformFogEnabled >= 0)
        {
            GL.Uniform1(_uniformFogEnabled, _fogEnabled ? 1 : 0);
        }
        if (_uniformLightingEnabled >= 0)
        {
            GL.Uniform1(_uniformLightingEnabled, _lightingEnabled ? 1 : 0);
        }

        foreach (var instance in _world.Objects)
        {
            if (!_modelResources.TryGetValue(instance.TypeId, out var resources))
            {
                continue;
            }

            var modelMatrix = BuildModelMatrix(instance);
            GL.UniformMatrix4(_uniformModel, false, ref modelMatrix);

            PrepareBonePalette(resources.Model);
            GL.UniformMatrix4(_uniformBones, MaxBones, false, ref _bonePalette[0].Row0.X);

            foreach (var mesh in resources.Meshes)
            {
                ApplyMaterial(mesh.Flags);

                GL.ActiveTexture(TextureUnit.Texture0);
                GL.BindTexture(TextureTarget.Texture2D, mesh.TextureHandle);
                GL.Uniform1(_uniformTexture, 0);
                GL.Uniform1(_uniformAlphaCutoff, mesh.Flags.HasFlag(MaterialFlags.AlphaTest) ? 0.35f : -1f);
                GL.Uniform1(_uniformMaterialFlags, unchecked((int)mesh.Flags));

                GL.BindVertexArray(mesh.VertexArrayObject);
                GL.DrawElements(PrimitiveType.Triangles, mesh.IndexCount, DrawElementsType.UnsignedInt, 0);
            }
        }

        ResetMaterialState();
    }

    public void Dispose()
    {
        ReleaseResources();
        if (_program != 0)
        {
            GL.DeleteProgram(_program);
            _program = 0;
        }
    }

    private void RebuildResources()
    {
        ReleaseResources();

        if (_world is null)
        {
            _hasAnimatedModels = false;
            _animationTime = 0f;
            _dirty = false;
            return;
        }

        EnsureProgram();

        var context = MapContext.ForMapId(_world.MapId);
        _textureLibrary = new TextureLibrary(_world.WorldPath, _world.ObjectDirectory, context);

        foreach (var entry in _world.ModelLibrary.Models)
        {
            var model = entry.Value;
            var meshes = new List<MeshResources>();
            foreach (var mesh in model.Meshes)
            {
                if (mesh.Positions.Length == 0 || mesh.Indices.Length == 0)
                {
                    continue;
                }
                meshes.Add(BuildMeshResources(mesh));
            }
            if (meshes.Count > 0)
            {
                _modelResources[entry.Key] = new ModelResources(model, meshes);
            }
        }

        _hasAnimatedModels = _modelResources.Values.Any(m => m.Model.Actions.Any(a => a.KeyframeCount > 1));
        _animationTime = 0f;
        _dirty = false;
    }

    private void EnsureProgram()
    {
        if (_program != 0)
        {
            return;
        }

        _program = ShaderFactory.CreateProgram(VertexShaderSource, FragmentShaderSource);
        _uniformModel = GL.GetUniformLocation(_program, "uModel");
        _uniformView = GL.GetUniformLocation(_program, "uView");
        _uniformProjection = GL.GetUniformLocation(_program, "uProjection");
        _uniformBones = GL.GetUniformLocation(_program, "uBones");
        _uniformLightDirection = GL.GetUniformLocation(_program, "uLightDirection");
        _uniformAlphaCutoff = GL.GetUniformLocation(_program, "uAlphaCutoff");
        _uniformMaterialFlags = GL.GetUniformLocation(_program, "uMaterialFlags");
        _uniformTexture = GL.GetUniformLocation(_program, "uTexture");
        _uniformCameraPosition = GL.GetUniformLocation(_program, "uCameraPosition");
        _uniformFogColor = GL.GetUniformLocation(_program, "uFogColor");
        _uniformFogParams = GL.GetUniformLocation(_program, "uFogParams");
        _uniformTime = GL.GetUniformLocation(_program, "uTime");
        _uniformFogEnabled = GL.GetUniformLocation(_program, "uFogEnabled");
        _uniformLightingEnabled = GL.GetUniformLocation(_program, "uLightingEnabled");
    }

    private void ReleaseResources()
    {
        foreach (var entry in _modelResources.Values)
        {
            foreach (var mesh in entry.Meshes)
            {
                if (mesh.VertexArrayObject != 0)
                {
                    GL.DeleteVertexArray(mesh.VertexArrayObject);
                }
                if (mesh.VertexBuffer != 0)
                {
                    GL.DeleteBuffer(mesh.VertexBuffer);
                }
                if (mesh.IndexBuffer != 0)
                {
                    GL.DeleteBuffer(mesh.IndexBuffer);
                }
            }
        }
        _modelResources.Clear();

        foreach (var handle in _textureCache.Values)
        {
            if (handle != 0)
            {
                GL.DeleteTexture(handle);
            }
        }
        _textureCache.Clear();
        _textureLibrary = null;
        _hasAnimatedModels = false;
        _animationTime = 0f;
    }

    private MeshResources BuildMeshResources(BmdMesh mesh)
    {
        var vertexCount = mesh.Positions.Length / 3;
        var stride = 9;
        var vertexData = new float[vertexCount * stride];

        for (var i = 0; i < vertexCount; i++)
        {
            var positionIndex = i * 3;
            var normalIndex = i * 3;
            var texIndex = i * 2;
            var baseOffset = i * stride;

            vertexData[baseOffset + 0] = mesh.Positions[positionIndex + 0];
            vertexData[baseOffset + 1] = mesh.Positions[positionIndex + 1];
            vertexData[baseOffset + 2] = mesh.Positions[positionIndex + 2];

            vertexData[baseOffset + 3] = mesh.Normals.Length > normalIndex + 2 ? mesh.Normals[normalIndex + 0] : 0f;
            vertexData[baseOffset + 4] = mesh.Normals.Length > normalIndex + 2 ? mesh.Normals[normalIndex + 1] : 1f;
            vertexData[baseOffset + 5] = mesh.Normals.Length > normalIndex + 2 ? mesh.Normals[normalIndex + 2] : 0f;

            vertexData[baseOffset + 6] = mesh.TexCoords.Length > texIndex + 1 ? mesh.TexCoords[texIndex + 0] : 0f;
            vertexData[baseOffset + 7] = mesh.TexCoords.Length > texIndex + 1 ? mesh.TexCoords[texIndex + 1] : 0f;

            var boneIndex = mesh.BoneIndices.Length > i ? mesh.BoneIndices[i] : (short)0;
            if (boneIndex < 0)
            {
                boneIndex = 0;
            }
            vertexData[baseOffset + 8] = boneIndex;
        }

        var vao = GL.GenVertexArray();
        var vbo = GL.GenBuffer();
        var ebo = GL.GenBuffer();

        GL.BindVertexArray(vao);

        GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
        GL.BufferData(BufferTarget.ArrayBuffer, vertexData.Length * sizeof(float), vertexData, BufferUsageHint.StaticDraw);

        GL.BindBuffer(BufferTarget.ElementArrayBuffer, ebo);
        GL.BufferData(BufferTarget.ElementArrayBuffer, mesh.Indices.Length * sizeof(uint), mesh.Indices, BufferUsageHint.StaticDraw);

        var strideBytes = stride * sizeof(float);
        GL.EnableVertexAttribArray(0);
        GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, strideBytes, 0);
        GL.EnableVertexAttribArray(1);
        GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, strideBytes, 3 * sizeof(float));
        GL.EnableVertexAttribArray(2);
        GL.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, strideBytes, 6 * sizeof(float));
        GL.EnableVertexAttribArray(3);
        GL.VertexAttribPointer(3, 1, VertexAttribPointerType.Float, false, strideBytes, 8 * sizeof(float));

        GL.BindVertexArray(0);

        var textureHandle = ResolveTexture(mesh.TextureName);

        return new MeshResources
        {
            VertexArrayObject = vao,
            VertexBuffer = vbo,
            IndexBuffer = ebo,
            IndexCount = mesh.Indices.Length,
            TextureHandle = textureHandle,
            Flags = mesh.MaterialFlags,
        };
    }

    private int ResolveTexture(string textureName)
    {
        var key = textureName;
        if (string.IsNullOrWhiteSpace(key))
        {
            key = "__default__";
        }

        if (_textureCache.TryGetValue(key, out var cached))
        {
            return cached;
        }

        TextureImage? image = null;
        if (_textureLibrary is not null)
        {
            (image, _) = _textureLibrary.LoadArbitrary(textureName);
            if (image is null && !string.IsNullOrWhiteSpace(textureName))
            {
                var stem = Path.GetFileNameWithoutExtension(textureName);
                if (!string.IsNullOrWhiteSpace(stem))
                {
                    (image, _) = _textureLibrary.LoadArbitrary(stem);
                }
            }
        }

        image ??= TextureImage.FromRgba(1, 1, new byte[] { 200, 200, 200, 255 });

        var handle = GL.GenTexture();
        GL.BindTexture(TextureTarget.Texture2D, handle);
        GL.PixelStore(PixelStoreParameter.UnpackAlignment, 1);
        GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, image.Width, image.Height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, image.Pixels);
        GlTextureHelpers.ApplySamplerParameters(TextureTarget.Texture2D, generateMipmaps: true);

        _textureCache[key] = handle;
        return handle;
    }

    private static Matrix4 BuildModelMatrix(ObjectInstance instance)
    {
        var scale = float.IsFinite(instance.Scale) && MathF.Abs(instance.Scale) > float.Epsilon ? instance.Scale : 1f;
        var model = Matrix4.Identity;
        model *= Matrix4.CreateScale(scale);

        var rx = MathHelper.DegreesToRadians(instance.Rotation.X);
        var ry = MathHelper.DegreesToRadians(instance.Rotation.Y);
        var rz = MathHelper.DegreesToRadians(instance.Rotation.Z);
        model *= Matrix4.CreateRotationX(rx);
        model *= Matrix4.CreateRotationY(ry);
        model *= Matrix4.CreateRotationZ(rz);
        model *= Matrix4.CreateTranslation(instance.Position.X, instance.Position.Y, instance.Position.Z);
        return model;
    }

    private void PrepareBonePalette(BmdModel model)
    {
        for (var i = 0; i < MaxBones; i++)
        {
            _bonePalette[i] = Matrix4.Identity;
        }

        var bones = model.Bones;
        if (bones.Count == 0)
        {
            return;
        }

        var actionIndex = 0;
        var hasAction = model.Actions.Count > actionIndex && model.Actions[actionIndex].KeyframeCount > 0;
        var action = hasAction ? model.Actions[actionIndex] : null;
        var keyframes = hasAction ? Math.Max(1, (int)action!.KeyframeCount) : 1;
        var frameTime = _animationTime * AnimationFps;
        var wrapped = keyframes > 0 ? frameTime % keyframes : 0f;
        var currentFrame = hasAction ? (int)MathF.Floor(wrapped) : 0;
        var nextFrame = hasAction ? (currentFrame + 1) % keyframes : 0;
        var t = hasAction ? wrapped - currentFrame : 0f;

        for (var boneIndex = 0; boneIndex < Math.Min(bones.Count, MaxBones); boneIndex++)
        {
            var bone = bones[boneIndex];
            if (bone.IsDummy)
            {
                _bonePalette[boneIndex] = bone.Parent >= 0 && bone.Parent < MaxBones
                    ? _bonePalette[bone.Parent]
                    : Matrix4.Identity;
                continue;
            }

            var local = Matrix4.Identity;
            if (!IsIdentity(bone.RestQuaternion))
            {
                local *= Matrix4.CreateFromQuaternion(ToQuaternion(bone.RestQuaternion));
            }
            if (!IsZero(bone.RestTranslation))
            {
                var rest = ToVector3(bone.RestTranslation);
                local *= Matrix4.CreateTranslation(rest.X, rest.Y, rest.Z);
            }

            if (hasAction && bone.Animations.Count > actionIndex)
            {
                var animation = bone.Animations[actionIndex];
                var position = InterpolatePosition(animation.Positions, currentFrame, nextFrame, t);
                var rotation = Interpolate(animation.Quaternions, currentFrame, nextFrame, t);

                local *= Matrix4.CreateFromQuaternion(ToQuaternion(rotation));
                var translated = ToVector3(position);
                local *= Matrix4.CreateTranslation(translated.X, translated.Y, translated.Z);
            }

            if (bone.Parent >= 0 && bone.Parent < MaxBones)
            {
                local = _bonePalette[bone.Parent] * local;
            }

            _bonePalette[boneIndex] = local;
        }

        if (hasAction && action!.LockPositions && action.LockedPositions.Count > 0)
        {
            var rootOffset = InterpolatePosition(action.LockedPositions, currentFrame, nextFrame, t);
            var offset = ToVector3(rootOffset);
            _bonePalette[0] *= Matrix4.CreateTranslation(offset.X, offset.Y, offset.Z);
        }
    }

    private static bool IsZero(System.Numerics.Vector3 value)
    {
        return Math.Abs(value.X) < float.Epsilon
            && Math.Abs(value.Y) < float.Epsilon
            && Math.Abs(value.Z) < float.Epsilon;
    }

    private static bool IsIdentity(System.Numerics.Quaternion value)
    {
        return Math.Abs(value.X) < float.Epsilon
            && Math.Abs(value.Y) < float.Epsilon
            && Math.Abs(value.Z) < float.Epsilon
            && Math.Abs(value.W - 1f) < float.Epsilon;
    }

    private static System.Numerics.Vector3 InterpolatePosition(IReadOnlyList<System.Numerics.Vector3> values, int current, int next, float t)
    {
        if (values.Count == 0)
        {
            return System.Numerics.Vector3.Zero;
        }
        var a = values[Math.Clamp(current, 0, values.Count - 1)];
        var b = values[Math.Clamp(next, 0, values.Count - 1)];
        return System.Numerics.Vector3.Lerp(a, b, t);
    }

    private static System.Numerics.Quaternion Interpolate(IReadOnlyList<System.Numerics.Quaternion> values, int current, int next, float t)
    {
        if (values.Count == 0)
        {
            return System.Numerics.Quaternion.Identity;
        }
        var a = values[Math.Clamp(current, 0, values.Count - 1)];
        var b = values[Math.Clamp(next, 0, values.Count - 1)];
        return System.Numerics.Quaternion.Slerp(a, b, t);
    }

    private static Quaternion ToQuaternion(System.Numerics.Quaternion value)
    {
        return new Quaternion(value.X, value.Y, value.Z, value.W);
    }

    private static Vector3 ToVector3(System.Numerics.Vector3 value)
    {
        return new Vector3(value.X, value.Y, value.Z);
    }

    private static void ApplyMaterial(MaterialFlags flags)
    {
        var transparent = flags.HasFlag(MaterialFlags.Transparent) || flags.HasFlag(MaterialFlags.Additive) || flags.HasFlag(MaterialFlags.Water) || flags.HasFlag(MaterialFlags.Lava);
        if (transparent)
        {
            GL.Enable(EnableCap.Blend);
            if (flags.HasFlag(MaterialFlags.Additive))
            {
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
            }
            else
            {
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            }
            GL.DepthMask(false);
        }
        else
        {
            GL.Disable(EnableCap.Blend);
            GL.DepthMask(true);
        }

        if (flags.HasFlag(MaterialFlags.DoubleSided))
        {
            GL.Disable(EnableCap.CullFace);
        }
        else
        {
            GL.Enable(EnableCap.CullFace);
            GL.CullFace(CullFaceMode.Back);
        }
    }

    private static void ResetMaterialState()
    {
        GL.Disable(EnableCap.Blend);
        GL.DepthMask(true);
        GL.Enable(EnableCap.CullFace);
        GL.CullFace(CullFaceMode.Back);
    }

    private sealed class ModelResources
    {
        public ModelResources(BmdModel model, List<MeshResources> meshes)
        {
            Model = model;
            Meshes = meshes;
        }

        public BmdModel Model { get; }
        public List<MeshResources> Meshes { get; }
    }

    private sealed class MeshResources
    {
        public int VertexArrayObject { get; init; }
        public int VertexBuffer { get; init; }
        public int IndexBuffer { get; init; }
        public int IndexCount { get; init; }
        public int TextureHandle { get; init; }
        public MaterialFlags Flags { get; init; }
    }

    private const string VertexShaderSource = @"#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in float aBoneIndex;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uBones[64];

out vec3 vNormal;
out vec2 vTexCoord;
out vec3 vWorldPos;

void main()
{
    int boneIndex = int(clamp(aBoneIndex + 0.5, 0.0, 63.0));
    mat4 boneMatrix = uBones[boneIndex];
    vec4 localPosition = boneMatrix * vec4(aPosition, 1.0);
    vec3 localNormal = mat3(boneMatrix) * aNormal;

    vec4 worldPosition = uModel * localPosition;
    vec3 worldNormal = mat3(uModel) * localNormal;

    vNormal = worldNormal;
    vTexCoord = aTexCoord;
    vWorldPos = worldPosition.xyz;
    gl_Position = uProjection * uView * worldPosition;
}";

    private const string FragmentShaderSource = @"#version 330 core
in vec3 vNormal;
in vec2 vTexCoord;
in vec3 vWorldPos;

uniform sampler2D uTexture;
uniform vec3 uLightDirection;
uniform float uAlphaCutoff;
uniform int uMaterialFlags;
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
    if ((uMaterialFlags & 1) != 0)
    {
        animatedUv += vec2(uTime * 0.02, uTime * 0.015);
    }
    else if ((uMaterialFlags & 2) != 0)
    {
        animatedUv += vec2(uTime * 0.01, -uTime * 0.02);
    }

    vec4 color = texture(uTexture, animatedUv);
    if (uAlphaCutoff > 0.0 && color.a < uAlphaCutoff)
    {
        discard;
    }

    vec3 lighting = color.rgb;
    if (uLightingEnabled != 0)
    {
        vec3 normal = normalize(vNormal);
        vec3 lightDir = normalize(-uLightDirection);
        vec3 viewDir = normalize(uCameraPosition - vWorldPos);
        vec3 halfDir = normalize(lightDir + viewDir);

        float diffuse = max(dot(normal, lightDir), 0.0);
        float ambient = 0.35;
        float emissiveBoost = ((uMaterialFlags & 16) != 0) ? 0.35 : 0.0;
        float specularWeight = ((uMaterialFlags & 3) != 0) ? 0.45 : 0.20;
        float shininess = ((uMaterialFlags & 1) != 0) ? 48.0 : 24.0;
        float specular = pow(max(dot(normal, halfDir), 0.0), shininess) * specularWeight;

        lighting = color.rgb * (ambient + diffuse * 0.65) + emissiveBoost * color.rgb + specular;
    }

    float fogFactor = 1.0;
    if (uFogEnabled != 0)
    {
        float distanceToCamera = length(uCameraPosition - vWorldPos);
        fogFactor = exp(-pow(distanceToCamera * uFogParams.x, uFogParams.y));
        fogFactor = clamp(fogFactor, 0.0, 1.0);
    }
    vec3 finalColor = mix(uFogColor, lighting, fogFactor);

    FragColor = vec4(finalColor, color.a);
}";
}
