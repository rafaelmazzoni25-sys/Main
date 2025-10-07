#pragma once

#include "terrain/TilePalette.h"
#include "terrain/Types.h"

#include <vector>

namespace terrain {

struct TerrainVertex {
    std::array<float, 3> position;
    std::array<float, 3> normal;
    std::array<float, 3> color;
};

class TerrainMesh {
public:
    TerrainMesh();

    void rebuild(const TerrainData& terrain, const TilePalette& palette);

    [[nodiscard]] const std::vector<TerrainVertex>& vertices() const noexcept { return m_vertices; }

private:
    std::vector<TerrainVertex> m_vertices;
};

}  // namespace terrain
