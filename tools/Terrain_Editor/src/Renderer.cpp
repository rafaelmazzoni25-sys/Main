#include "terrain/Renderer.h"

#include "terrain/Types.h"

#include <GLFW/glfw3.h>
#include <GL/glu.h>

#include <algorithm>
#include <array>

namespace terrain {

Renderer::Renderer(GLFWwindow* window)
    : m_window(window) {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_COLOR_MATERIAL);
}

Renderer::~Renderer() = default;

void Renderer::setWorldData(const WorldData& world) {
    m_world = &world;
    m_mesh.rebuild(world.terrain, m_palette);
}

void Renderer::updateCamera(double deltaSeconds) {
    if (!m_window) {
        return;
    }

    m_camera.update(deltaSeconds);

    double cursorX = 0.0;
    double cursorY = 0.0;
    glfwGetCursorPos(m_window, &cursorX, &cursorY);

    const bool leftPressed = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    const bool middlePressed = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

    if (leftPressed && !m_rotating) {
        m_rotating = true;
        m_lastCursorX = cursorX;
        m_lastCursorY = cursorY;
    } else if (!leftPressed) {
        m_rotating = false;
    }

    if (middlePressed && !m_panning) {
        m_panning = true;
        m_lastCursorX = cursorX;
        m_lastCursorY = cursorY;
    } else if (!middlePressed) {
        m_panning = false;
    }

    if (m_rotating || m_panning) {
        const double dx = cursorX - m_lastCursorX;
        const double dy = cursorY - m_lastCursorY;
        m_lastCursorX = cursorX;
        m_lastCursorY = cursorY;

        if (m_rotating) {
            const float yaw = static_cast<float>(dx) * 0.005f;
            const float pitch = static_cast<float>(dy) * 0.005f;
            m_camera.orbit(-yaw, -pitch);
        }
        if (m_panning) {
            const float panSpeed = 4.0f * kTerrainScale;
            m_camera.pan(static_cast<float>(-dx) * panSpeed * static_cast<float>(deltaSeconds),
                         static_cast<float>(dy) * panSpeed * static_cast<float>(deltaSeconds));
        }
    }

    if (glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS) {
        m_camera.zoom(-2000.0f * static_cast<float>(deltaSeconds));
    }
    if (glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS) {
        m_camera.zoom(2000.0f * static_cast<float>(deltaSeconds));
    }
}

void Renderer::render() {
    if (!m_window || !m_world) {
        return;
    }

    glClearColor(0.55f, 0.68f, 0.82f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    configureProjection();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    const auto camPos = m_camera.position();
    const auto camTarget = m_camera.target();
    gluLookAt(camPos[0], camPos[1], camPos[2],
              camTarget[0], camTarget[1], camTarget[2],
              0.0f, 1.0f, 0.0f);

    const GLfloat lightPosition[] = {-0.5f, 1.0f, -0.4f, 0.0f};
    const GLfloat ambient[] = {0.25f, 0.25f, 0.25f, 1.0f};
    const GLfloat diffuse[] = {0.85f, 0.85f, 0.85f, 1.0f};
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);

    renderTerrain();

    glDisable(GL_LIGHTING);
    renderObjects();
}

void Renderer::configureProjection() const {
    int width = 1;
    int height = 1;
    glfwGetFramebufferSize(m_window, &width, &height);
    const float aspect = height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
    gluPerspective(60.0f, aspect, 100.0f, 60000.0f);
}

void Renderer::renderTerrain() const {
    const auto& vertices = m_mesh.vertices();
    if (vertices.empty()) {
        return;
    }

    glBegin(GL_TRIANGLES);
    for (const auto& vertex : vertices) {
        glColor3fv(vertex.color.data());
        glNormal3fv(vertex.normal.data());
        glVertex3fv(vertex.position.data());
    }
    glEnd();
}

void Renderer::renderObjects() const {
    if (!m_world) {
        return;
    }

    glLineWidth(1.5f);
    glBegin(GL_LINES);
    for (const auto& object : m_world->objects) {
        const float size = 150.0f * std::max(0.2f, object.scale);
        const float x = object.position[0];
        const float y = object.position[1];
        const float z = object.position[2];

        glColor3f(0.9f, 0.2f, 0.1f);
        glVertex3f(x - size, y, z);
        glVertex3f(x + size, y, z);

        glColor3f(0.2f, 0.9f, 0.1f);
        glVertex3f(x, y - size * 0.5f, z);
        glVertex3f(x, y + size * 0.5f, z);

        glColor3f(0.1f, 0.4f, 0.9f);
        glVertex3f(x, y, z - size);
        glVertex3f(x, y, z + size);
    }
    glEnd();
}

}  // namespace terrain
