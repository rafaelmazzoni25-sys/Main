cmake_minimum_required(VERSION 3.16)
project(Terrain_Editor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

set(TERRAIN_EDITOR_SOURCES
    src/Application.cpp
    src/Camera.cpp
    src/Renderer.cpp
    src/TerrainMesh.cpp
    src/TilePalette.cpp
    src/WorldLoader.cpp
    src/main.cpp
)

add_executable(terrain_editor ${TERRAIN_EDITOR_SOURCES})

target_include_directories(terrain_editor
    PRIVATE
        include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/include
)

target_link_libraries(terrain_editor
    PRIVATE
        glfw
        OpenGL::GL
        OpenGL::GLU
)

target_compile_features(terrain_editor PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(terrain_editor PRIVATE /permissive- /W4 /WX-)
else()
    target_compile_options(terrain_editor PRIVATE -Wall -Wextra -Wpedantic)
endif()
