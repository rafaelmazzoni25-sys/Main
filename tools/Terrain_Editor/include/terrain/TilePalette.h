#pragma once

#include <array>
#include <cstdint>

namespace terrain {

struct TilePalette {
    static constexpr std::size_t kColorCount = 256;

    TilePalette();

    [[nodiscard]] std::array<float, 3> colorForTile(std::uint8_t tileIndex) const noexcept;

private:
    std::array<std::array<float, 3>, kColorCount> m_colors;
};

}  // namespace terrain
