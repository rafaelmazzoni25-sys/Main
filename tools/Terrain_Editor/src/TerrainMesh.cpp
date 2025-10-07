#include "terrain/TerrainMesh.h"

#include "terrain/Types.h"

#include <glm/glm.hpp>

#include <algorithm>

namespace terrain {

namespace {
float sampleHeight(const TerrainData& data, int x, int y) {
    x = std::clamp(x, 0, static_cast<int>(kTerrainSize) - 1);
    y = std::clamp(y, 0, static_cast<int>(kTerrainSize) - 1);
    return data.height[y * kTerrainSize + x];
}

glm::vec3 computeNormal(const TerrainData& data, int x, int y) {
    const float hL = sampleHeight(data, x - 1, y);
    const float hR = sampleHeight(data, x + 1, y);
    const float hD = sampleHeight(data, x, y - 1);
    const float hU = sampleHeight(data, x, y + 1);

    glm::vec3 normal{
        hL - hR,
        2.0f * kTerrainScale,
        hD - hU,
    };
    return glm::normalize(normal);
}

glm::vec3 makePosition(int x, int y, float height) {
    return glm::vec3{
        static_cast<float>(x) * kTerrainScale,
        height,
        static_cast<float>(y) * kTerrainScale,
    };
}

std::array<float, 3> toArray(const glm::vec3& value) {
    return {value.x, value.y, value.z};
}
}

TerrainMesh::TerrainMesh() = default;

void TerrainMesh::rebuild(const TerrainData& terrain, const TilePalette& palette) {
    m_vertices.clear();
    if (terrain.empty()) {
        return;
    }

    m_vertices.reserve((kTerrainSize - 1) * (kTerrainSize - 1) * 6);
    for (std::size_t y = 0; y < kTerrainSize - 1; ++y) {
        for (std::size_t x = 0; x < kTerrainSize - 1; ++x) {
            const std::size_t idx00 = y * kTerrainSize + x;
            const std::size_t idx10 = y * kTerrainSize + (x + 1);
            const std::size_t idx01 = (y + 1) * kTerrainSize + x;
            const std::size_t idx11 = (y + 1) * kTerrainSize + (x + 1);

            const glm::vec3 p00 = makePosition(static_cast<int>(x), static_cast<int>(y), terrain.height[idx00]);
            const glm::vec3 p10 = makePosition(static_cast<int>(x + 1), static_cast<int>(y), terrain.height[idx10]);
            const glm::vec3 p01 = makePosition(static_cast<int>(x), static_cast<int>(y + 1), terrain.height[idx01]);
            const glm::vec3 p11 = makePosition(static_cast<int>(x + 1), static_cast<int>(y + 1), terrain.height[idx11]);

            const glm::vec3 n00 = computeNormal(terrain, static_cast<int>(x), static_cast<int>(y));
            const glm::vec3 n10 = computeNormal(terrain, static_cast<int>(x + 1), static_cast<int>(y));
            const glm::vec3 n01 = computeNormal(terrain, static_cast<int>(x), static_cast<int>(y + 1));
            const glm::vec3 n11 = computeNormal(terrain, static_cast<int>(x + 1), static_cast<int>(y + 1));

            const auto colorPrimary = palette.colorForTile(terrain.layer1[idx00]);

            TerrainVertex tri1_v1{toArray(p00), toArray(n00), colorPrimary};
            TerrainVertex tri1_v2{toArray(p10), toArray(n10), colorPrimary};
            TerrainVertex tri1_v3{toArray(p11), toArray(n11), colorPrimary};
            m_vertices.emplace_back(tri1_v1);
            m_vertices.emplace_back(tri1_v2);
            m_vertices.emplace_back(tri1_v3);

            TerrainVertex tri2_v1{toArray(p00), toArray(n00), colorPrimary};
            TerrainVertex tri2_v2{toArray(p11), toArray(n11), colorPrimary};
            TerrainVertex tri2_v3{toArray(p01), toArray(n01), colorPrimary};
            m_vertices.emplace_back(tri2_v1);
            m_vertices.emplace_back(tri2_v2);
            m_vertices.emplace_back(tri2_v3);
        }
    }
}

}  // namespace terrain
