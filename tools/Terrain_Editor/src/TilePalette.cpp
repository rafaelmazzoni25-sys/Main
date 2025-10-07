#include "terrain/TilePalette.h"

#include <cmath>

namespace terrain {

namespace {
struct TileColorEntry {
    std::uint8_t tile;
    std::array<float, 3> color;
};

constexpr TileColorEntry kBaseColors[] = {
    {0, {0.35f, 0.52f, 0.28f}},
    {1, {0.40f, 0.60f, 0.32f}},
    {2, {0.55f, 0.48f, 0.33f}},
    {3, {0.60f, 0.50f, 0.36f}},
    {4, {0.62f, 0.52f, 0.38f}},
    {5, {0.20f, 0.35f, 0.55f}},
    {6, {0.50f, 0.40f, 0.28f}},
    {7, {0.42f, 0.42f, 0.42f}},
    {8, {0.35f, 0.35f, 0.38f}},
    {9, {0.45f, 0.32f, 0.25f}},
    {10, {0.50f, 0.45f, 0.40f}},
    {11, {0.65f, 0.24f, 0.18f}},
    {12, {0.55f, 0.36f, 0.28f}},
    {13, {0.30f, 0.30f, 0.30f}},
};

std::array<float, 3> makeHashedColor(std::uint8_t tile) {
    const float t = static_cast<float>(tile) / 255.0f;
    const float r = 0.25f + std::fmod(std::sin(t * 91.0f) * 0.5f + 0.5f, 1.0f) * 0.7f;
    const float g = 0.25f + std::fmod(std::sin((t + 0.33f) * 83.0f) * 0.5f + 0.5f, 1.0f) * 0.7f;
    const float b = 0.25f + std::fmod(std::sin((t + 0.66f) * 79.0f) * 0.5f + 0.5f, 1.0f) * 0.7f;
    return {r, g, b};
}
}  // namespace

TilePalette::TilePalette() {
    for (std::size_t i = 0; i < kColorCount; ++i) {
        m_colors[i] = makeHashedColor(static_cast<std::uint8_t>(i));
    }
    for (const auto& entry : kBaseColors) {
        m_colors[entry.tile] = entry.color;
    }
}

std::array<float, 3> TilePalette::colorForTile(std::uint8_t tileIndex) const noexcept {
    return m_colors[tileIndex];
}

}  // namespace terrain
