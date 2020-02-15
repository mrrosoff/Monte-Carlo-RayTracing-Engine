# Created by Max Rosoff on 10/19/2019.

# CMake Requirements and Setup
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(RayTracer)
set(CMAKE_CXX_STANDARD 14)

option(USE_CUDA "Use CUDA" OFF)

file(GLOB cpu_main_file "${CMAKE_SOURCE_DIR}/RayTracingFiles/CPU/main.cpp")
file(GLOB_RECURSE cpu_source_files "${CMAKE_SOURCE_DIR}/RayTracingFiles/CPU/*.cpp")
file(GLOB_RECURSE cpu_header_files "${CMAKE_SOURCE_DIR}/RayTracingFiles/CPU/*.h")
file(GLOB_RECURSE gpu_source_files "${CMAKE_SOURCE_DIR}/RayTracingFiles/GPU/*.cu")
file(GLOB_RECURSE gpu_header_files "${CMAKE_SOURCE_DIR}/RayTracingFiles/GPU/*.h")

if(USE_CUDA)

    enable_language("CUDA")
    add_compile_options(-O3)
    add_executable(raytracer ${gpu_source_files} ${gpu_header_files})

else(USE_CUDA)

    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()

    add_compile_options(-Wall -O3)
    add_executable(raytracer ${cpu_main_file} ${cpu_source_files} ${cpu_header_files})

endif()

if(USE_CUDA)

    add_definitions(-DUSE_CUDA)

endif()
