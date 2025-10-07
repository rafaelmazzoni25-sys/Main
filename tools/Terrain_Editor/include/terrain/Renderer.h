#pragma once

#include "terrain/Camera.h"
#include "terrain/TerrainMesh.h"
#include "terrain/Types.h"

#include <memory>
#include <optional>

struct GLFWwindow;

namespace terrain {

class Renderer {
public:
    explicit Renderer(GLFWwindow* window);
    ~Renderer();

    void setWorldData(const WorldData& world);
    void updateCamera(double deltaSeconds);
    void render();

    OrbitCamera& camera() noexcept { return m_camera; }

private:
    GLFWwindow* m_window;
    OrbitCamera m_camera;
    TerrainMesh m_mesh;
    TilePalette m_palette;
    const WorldData* m_world = nullptr;
    bool m_rotating = false;
    bool m_panning = false;
    double m_lastCursorX = 0.0;
    double m_lastCursorY = 0.0;

    void configureProjection() const;
    void renderTerrain() const;
    void renderObjects() const;
};

}  // namespace terrain
