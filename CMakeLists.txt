cmake_minimum_required(VERSION 3.7.1)

set(NAME VulkanCompute)

project(${NAME})

set(CMAKE_CXX_STANDARD 17)

include_directories(external/glm)

find_package(Vulkan REQUIRED)

find_package(X11 REQUIRED)
link_libraries(${X11_LIBRARIES})
add_definitions(-DVK_USE_PLATFORM_XLIB_KHR)

add_executable(${NAME} main.cpp)
target_link_libraries(${NAME} Vulkan::Vulkan)