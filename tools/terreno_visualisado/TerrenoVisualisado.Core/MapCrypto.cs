namespace TerrenoVisualisado.Core;

internal static class MapCrypto
{
    private static readonly byte[] XorKey =
    {
        0xD1, 0x73, 0x52, 0xF6, 0xD2, 0x9A, 0xCB, 0x27,
        0x3E, 0xAF, 0x59, 0x31, 0x37, 0xB3, 0xE7, 0xA2,
    };

    public static byte[] Decrypt(ReadOnlySpan<byte> input)
    {
        var output = new byte[input.Length];
        byte wMapKey = 0x5E;
        for (var i = 0; i < input.Length; i++)
        {
            var value = input[i];
            var decrypted = (byte)(((value ^ XorKey[i % XorKey.Length]) - wMapKey) & 0xFF);
            output[i] = decrypted;
            wMapKey = (byte)((value + 0x3D) & 0xFF);
        }
        return output;
    }
}
