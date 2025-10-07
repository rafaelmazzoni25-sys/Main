#pragma once

#include "terrain/Types.h"

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace terrain {

class WorldLoader {
public:
    WorldLoader();

    WorldData load(const std::filesystem::path& worldDirectory, const LoadOptions& options) const;

private:
    using ModelNameTable = std::map<int, std::string>;

    static ModelNameTable loadModelNames(const std::optional<std::filesystem::path>& enumPath);
    static std::optional<int> inferMapId(const std::filesystem::path& worldDirectory);
    static std::filesystem::path resolveAttributesPath(const std::filesystem::path& worldDirectory, std::optional<int> mapId);
    static std::filesystem::path resolveMappingPath(const std::filesystem::path& worldDirectory, std::optional<int> mapId);
    static std::filesystem::path resolveObjectsPath(
        const std::filesystem::path& worldDirectory,
        const std::optional<std::filesystem::path>& objectRoot,
        std::optional<int> mapId);
    static std::filesystem::path resolveHeightPath(const std::filesystem::path& worldDirectory, bool preferExtended);

    static TerrainData loadTerrain(
        const std::filesystem::path& attributesPath,
        const std::filesystem::path& mappingPath,
        const std::filesystem::path& heightPath,
        bool forceExtendedHeight,
        std::optional<float> heightScale,
        int& outAttributeMapId,
        int& outMappingMapId);

    static std::vector<ObjectInstance> loadObjects(
        const std::filesystem::path& objectsPath,
        const ModelNameTable& modelNames,
        int& outVersion,
        int& outMapId);
};

}  // namespace terrain
