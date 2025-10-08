using System;
using System.Collections.Generic;

namespace TerrenoVisualisado.Core;

public readonly struct MapContext
{
    public const int InvalidMapId = -1;

    public MapContext(int mapId)
    {
        MapId = mapId;
    }

    public int MapId { get; }

    public bool HasValidMapId => MapId >= 0;

    public static MapContext ForMapId(int? mapId)
    {
        return new MapContext(mapId ?? InvalidMapId);
    }

    public bool IsBattleCastle => MapId == (int)MapIdConstants.BattleCastle;
    public bool IsCryWolf => MapId == (int)MapIdConstants.CryWolfFirst;
    public bool IsKanturuThird => MapId == (int)MapIdConstants.KanturuThird;
    public bool IsPkField => MapId == (int)MapIdConstants.PkField;
    public bool IsDoppelGanger2 => MapId == (int)MapIdConstants.DoppelGanger2;
    public bool IsKarutan => MapId == (int)MapIdConstants.Karutan1 || MapId == (int)MapIdConstants.Karutan2;
    public bool IsCursedTemple => MapId >= (int)MapIdConstants.CursedTempleLevel1 && MapId <= (int)MapIdConstants.CursedTempleLevel6;
    public bool IsHome6thCharacter => MapId == (int)MapIdConstants.Home6thCharacter;
    public bool IsNewLoginScene => MapId == (int)MapIdConstants.NewLoginScene73 || MapId == (int)MapIdConstants.NewLoginScene77;
    public bool IsNewCharacterScene => MapId == (int)MapIdConstants.NewCharacterScene74 || MapId == (int)MapIdConstants.NewCharacterScene78;
    public bool IsEmpireGuardian => MapId >= (int)MapIdConstants.EmpireGuardian1 && MapId <= (int)MapIdConstants.EmpireGuardian4 || IsNewLoginScene || IsNewCharacterScene;

    public bool HasWaterTerrain => MapId >= (int)MapIdConstants.Hellas && MapId <= (int)MapIdConstants.HellasEnd || MapId == (int)MapIdConstants.HellasHidden;

    public IEnumerable<string> EnumerateTerrainLightCandidates()
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        return Enumerate();

        IEnumerable<string> Enumerate()
        {
            if (IsBattleCastle && seen.Add("TerrainLight2"))
            {
                yield return "TerrainLight2";
            }

            if (seen.Add("TerrainLight"))
            {
                yield return "TerrainLight";
            }

            if (IsCryWolf && seen.Add("TerrainLight1"))
            {
                yield return "TerrainLight1";
            }

            if (IsCryWolf && seen.Add("TerrainLight2"))
            {
                yield return "TerrainLight2";
            }

            if (seen.Add("TerrainLight1"))
            {
                yield return "TerrainLight1";
            }

            if (seen.Add("TerrainLight2"))
            {
                yield return "TerrainLight2";
            }
        }
    }

    public bool UsesAlphaGround01 => IsHome6thCharacter || IsNewLoginScene || IsNewCharacterScene;
    public bool UsesAlphaTileForRock04 => IsEmpireGuardian || IsNewLoginScene || IsNewCharacterScene;

    private enum MapIdConstants
    {
        Hellas = 24,
        HellasEnd = Hellas + 5,
        HellasHidden = 36,
        BattleCastle = 30,
        CryWolfFirst = 34,
        KanturuThird = 39,
        CursedTempleLevel1 = 45,
        CursedTempleLevel6 = 50,
        Home6thCharacter = 51,
        NewLoginScene73 = 73,
        NewCharacterScene74 = 74,
        NewLoginScene77 = 77,
        NewCharacterScene78 = 78,
        PkField = 63,
        DoppelGanger2 = 66,
        EmpireGuardian1 = 69,
        EmpireGuardian2 = 70,
        EmpireGuardian3 = 71,
        EmpireGuardian4 = 72,
        Karutan1 = 80,
        Karutan2 = 81,
    }
}
