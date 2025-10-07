#include "terrain/Application.h"

#include "terrain/Renderer.h"
#include "terrain/WorldLoader.h"

#include <GLFW/glfw3.h>

#include <chrono>
#include <iostream>

namespace terrain {

namespace {
constexpr int kDefaultWidth = 1280;
constexpr int kDefaultHeight = 720;
}

Application::Application() = default;
Application::~Application() {
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

bool Application::initialize() {
    if (!glfwInit()) {
        std::cerr << "Falha ao inicializar GLFW." << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    m_window = glfwCreateWindow(kDefaultWidth, kDefaultHeight, "Terrain_Editor", nullptr, nullptr);
    if (!m_window) {
        std::cerr << "Não foi possível criar a janela do Terrain_Editor." << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);

    m_renderer = std::make_unique<Renderer>(m_window);
    return true;
}

bool Application::loadWorld(const std::filesystem::path& worldDirectory, const LoadOptions& options) {
    try {
        m_world = m_loader.load(worldDirectory, options);
        if (m_renderer) {
            m_renderer->setWorldData(m_world);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Falha ao carregar o mundo: " << ex.what() << std::endl;
        return false;
    }
    return true;
}

void Application::run() {
    if (!m_window || !m_renderer) {
        return;
    }

    m_running = true;
    auto lastTime = std::chrono::steady_clock::now();
    while (m_running && !glfwWindowShouldClose(m_window)) {
        auto now = std::chrono::steady_clock::now();
        double deltaSeconds = std::chrono::duration<double>(now - lastTime).count();
        lastTime = now;

        glfwPollEvents();
        m_renderer->updateCamera(deltaSeconds);
        m_renderer->render();
        glfwSwapBuffers(m_window);
    }
}

void Application::framebufferSizeCallback(GLFWwindow* /*window*/, int width, int height) {
    glViewport(0, 0, width, height);
}

}  // namespace terrain
