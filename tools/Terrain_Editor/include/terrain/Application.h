#pragma once

#include "terrain/Renderer.h"
#include "terrain/Types.h"
#include "terrain/WorldLoader.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

struct GLFWwindow;

namespace terrain {

class Application {
public:
    Application();
    ~Application();

    bool initialize();
    bool loadWorld(const std::filesystem::path& worldDirectory, const LoadOptions& options);
    void run();

private:
    GLFWwindow* m_window = nullptr;
    std::unique_ptr<Renderer> m_renderer;
    WorldLoader m_loader;
    WorldData m_world;
    bool m_running = false;

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};

}  // namespace terrain
