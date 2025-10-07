#include "terrain/Camera.h"

#include "terrain/Types.h"

#include <algorithm>
#include <numbers>

namespace terrain {

OrbitCamera::OrbitCamera()
    : m_target{static_cast<float>(kTerrainSize / 2 * kTerrainScale),
               0.0f,
               static_cast<float>(kTerrainSize / 2 * kTerrainScale)},
      m_distance(9000.0f),
      m_yaw(std::numbers::pi_v<float> * 1.25f),
      m_pitch(std::numbers::pi_v<float> / 4.0f),
      m_minPitch(std::numbers::pi_v<float> / 180.0f * 5.0f),
      m_maxPitch(std::numbers::pi_v<float> / 180.0f * 85.0f),
      m_minDistance(1000.0f),
      m_maxDistance(30000.0f) {}

void OrbitCamera::update(double /*deltaSeconds*/) {
    m_pitch = std::clamp(m_pitch, m_minPitch, m_maxPitch);
    m_distance = std::clamp(m_distance, m_minDistance, m_maxDistance);
}

void OrbitCamera::orbit(float deltaYaw, float deltaPitch) {
    m_yaw += deltaYaw;
    m_pitch = std::clamp(m_pitch + deltaPitch, m_minPitch, m_maxPitch);
}

void OrbitCamera::zoom(float deltaDistance) {
    m_distance = std::clamp(m_distance + deltaDistance, m_minDistance, m_maxDistance);
}

void OrbitCamera::pan(float offsetX, float offsetY) {
    m_target[0] += offsetX;
    m_target[2] += offsetY;
}

std::array<float, 3> OrbitCamera::position() const noexcept {
    const float cosPitch = std::cos(m_pitch);
    const float sinPitch = std::sin(m_pitch);
    const float cosYaw = std::cos(m_yaw);
    const float sinYaw = std::sin(m_yaw);

    const float dirX = cosPitch * cosYaw;
    const float dirY = sinPitch;
    const float dirZ = cosPitch * sinYaw;

    return {
        m_target[0] - dirX * m_distance,
        m_target[1] - dirY * m_distance,
        m_target[2] - dirZ * m_distance,
    };
}

}  // namespace terrain
