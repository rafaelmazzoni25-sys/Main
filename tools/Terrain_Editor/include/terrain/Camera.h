#pragma once

#include <array>

namespace terrain {

class OrbitCamera {
public:
    OrbitCamera();

    void update(double deltaSeconds);
    void orbit(float deltaYaw, float deltaPitch);
    void zoom(float deltaDistance);
    void pan(float offsetX, float offsetY);

    [[nodiscard]] std::array<float, 3> position() const noexcept;
    [[nodiscard]] std::array<float, 3> target() const noexcept { return m_target; }
    [[nodiscard]] float distance() const noexcept { return m_distance; }

private:
    std::array<float, 3> m_target;
    float m_distance;
    float m_yaw;
    float m_pitch;
    float m_minPitch;
    float m_maxPitch;
    float m_minDistance;
    float m_maxDistance;
};

}  // namespace terrain
