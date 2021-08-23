cmake_minimum_required(VERSION 3.7.1)

set(NAME VulkanCompute)

project(${NAME})

set(CMAKE_CXX_STANDARD 17)

include_directories(external/glm)
include_directories(external/stb)

find_package(Vulkan REQUIRED)

IF(WIN32)
	add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
	add_definitions(-DVK_KHR_SURFACE_EXTENSION_NAME)
	add_definitions(-DVK_KHR_WIN32_SURFACE_EXTENSION_NAME)
ELSE()
	find_package(X11 REQUIRED)
	link_libraries(${X11_LIBRARIES})
	add_definitions(-DVK_KHR_SURFACE_EXTENSION_NAME)
	add_definitions(-DVK_KHR_XLIB_SURFACE_EXTENSION_NAME)
	add_definitions(-DVK_USE_PLATFORM_XLIB_KHR)
ENDIF()

add_executable(${NAME} main.cpp)
target_link_libraries(${NAME} Vulkan::Vulkan ${WINLIBS})