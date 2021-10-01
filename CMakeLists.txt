# CMake Requirements and Setup

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(RayTracer)
set(CMAKE_CXX_STANDARD 17)

file(GLOB main "${CMAKE_SOURCE_DIR}/RayTracingFiles/main.cpp")
file(GLOB_RECURSE file_in_out "${CMAKE_SOURCE_DIR}/RayTracingFiles/FileInOut/*.cpp")
file(GLOB_RECURSE matrix "${CMAKE_SOURCE_DIR}/RayTracingFiles/Matrix/*.cpp")
file(GLOB_RECURSE scene_items "${CMAKE_SOURCE_DIR}/RayTracingFiles/SceneItems/*.cpp")

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_compile_options(-Wall -O3)
add_executable(raytracer 
    ${main} 
    ${file_in_out} 
    ${matrix} 
    ${scene_items} 
    "${CMAKE_SOURCE_DIR}/RayTracingFiles/RayTracing/Camera.cpp" 
    "${CMAKE_SOURCE_DIR}/RayTracingFiles/RayTracing/Ray.cpp" 
    "${CMAKE_SOURCE_DIR}/RayTracingFiles/RayTracing/RayTracer.cpp"
)