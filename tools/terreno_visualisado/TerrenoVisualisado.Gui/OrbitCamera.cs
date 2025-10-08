using OpenTK.Mathematics;

namespace TerrenoVisualisado.Gui;

internal sealed class OrbitCamera
{
    public Vector3 Target { get; set; } = Vector3.Zero;
    public float Distance { get; set; } = 2000f;
    public float Azimuth { get; set; } = MathHelper.DegreesToRadians(45f);
    public float Elevation { get; set; } = MathHelper.DegreesToRadians(35f);

    public Vector3 Position
    {
        get
        {
            var offset = GetOffset();
            return Target + offset * Distance;
        }
    }

    public Vector3 Forward
    {
        get
        {
            var forward = Target - Position;
            if (forward.LengthSquared <= float.Epsilon)
            {
                return -Vector3.UnitZ;
            }
            return Vector3.Normalize(forward);
        }
    }

    public Vector3 Right
    {
        get
        {
            var forward = Forward;
            var right = Vector3.Cross(forward, Vector3.UnitY);
            if (right.LengthSquared <= float.Epsilon)
            {
                return Vector3.UnitX;
            }
            return Vector3.Normalize(right);
        }
    }

    public Vector3 Up
    {
        get
        {
            var right = Right;
            var forward = Forward;
            var up = Vector3.Cross(right, forward);
            if (up.LengthSquared <= float.Epsilon)
            {
                return Vector3.UnitY;
            }
            return Vector3.Normalize(up);
        }
    }

    public Matrix4 ViewMatrix => Matrix4.LookAt(Position, Target, Vector3.UnitY);

    public Matrix4 ProjectionMatrix(float aspectRatio)
    {
        var clampedAspect = Math.Max(0.01f, aspectRatio);
        return Matrix4.CreatePerspectiveFieldOfView(MathHelper.DegreesToRadians(60f), clampedAspect, 10f, 200000f);
    }

    private Vector3 GetOffset()
    {
        var cosElevation = MathF.Cos(Elevation);
        var sinElevation = MathF.Sin(Elevation);
        var cosAzimuth = MathF.Cos(Azimuth);
        var sinAzimuth = MathF.Sin(Azimuth);
        return new Vector3(cosElevation * cosAzimuth, sinElevation, cosElevation * sinAzimuth);
    }
}
