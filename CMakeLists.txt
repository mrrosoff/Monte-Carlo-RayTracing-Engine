# CMake Requirements and Setup

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(RayTracer)
set(CMAKE_CXX_STANDARD 17)

file(GLOB cpu_main_file "${CMAKE_SOURCE_DIR}/RayTracingFiles/main.cpp")
file(GLOB_RECURSE cpu_source_files "${CMAKE_SOURCE_DIR}/RayTracingFiles/*.cpp")

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_compile_options(-Wall -O3)
add_executable(raytracer ${cpu_main_file} ${cpu_source_files})