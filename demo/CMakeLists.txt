cmake_minimum_required(VERSION 3.16)
project(vulkan_x11_cube LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Vulkan REQUIRED)


# Find GLFW (pkg-config fallback if CMake config not found)
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(GLFW3 QUIET glfw3)
endif()


if(GLFW3_FOUND)
    include_directories(${GLFW3_INCLUDE_DIRS})
    link_directories(${GLFW3_LIBRARY_DIRS})
    set(GLFW3_LIBRARIES ${GLFW3_LIBRARIES})
else()
    # Try standard CMake package
    find_package(glfw3 REQUIRED)
    set(GLFW3_LIBRARIES glfw)
endif()


add_executable(vulkan_x11_cube
    src/main.cpp
)


target_include_directories(vulkan_x11_cube PRIVATE ${Vulkan_INCLUDE_DIRS})


target_link_libraries(vulkan_x11_cube PRIVATE ${Vulkan_LIBRARIES} ${GLFW3_LIBRARIES})


# Copy SPIR-V to build dir for runtime loading
add_custom_command(TARGET vulkan_x11_cube POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:vulkan_x11_cube>/shaders/bin"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/shaders/bin/cube.vert.spv"
    "$<TARGET_FILE_DIR:vulkan_x11_cube>/shaders/bin/cube.vert.spv"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/shaders/bin/cube.frag.spv"
    "$<TARGET_FILE_DIR:vulkan_x11_cube>/shaders/bin/cube.frag.spv"
)
