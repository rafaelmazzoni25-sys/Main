#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace terrain {

constexpr std::size_t kTerrainSize = 256;
constexpr float kTerrainScale = 100.0f;

struct TerrainData {
    std::vector<float> height;              // size = kTerrainSize * kTerrainSize
    std::vector<std::uint8_t> layer1;       // tile index for primary layer
    std::vector<std::uint8_t> layer2;       // tile index for secondary layer
    std::vector<float> alpha;               // blend factor between layers
    std::vector<std::uint16_t> attributes;  // tile attributes

    bool empty() const noexcept { return height.empty(); }
};

struct ObjectInstance {
    std::int16_t typeId = 0;
    std::array<float, 3> position{0.0f, 0.0f, 0.0f};
    std::array<float, 3> rotation{0.0f, 0.0f, 0.0f};
    float scale = 1.0f;
    std::optional<std::string> typeName{};
};

struct WorldData {
    std::filesystem::path worldPath;
    std::filesystem::path objectsPath;
    int mapId = -1;
    int objectVersion = 0;
    TerrainData terrain;
    std::vector<ObjectInstance> objects;
};

struct LoadOptions {
    std::optional<int> mapId;
    std::optional<std::filesystem::path> objectRoot;
    bool forceExtendedHeight = false;
    std::optional<float> heightScale;
    std::optional<std::filesystem::path> enumPath;
};

}  // namespace terrain
