# Created by Max Rosoff on 10/19/2019.

# CMake Requirements and Setup
cmake_minimum_required(VERSION 3.10)
project(RayTracer)
set(CMAKE_CXX_STANDARD 14)

# Compile Flags
add_compile_options(-Wall -O3)

# Parallel Capabilities
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Include Matrix Library (Eigen)
include_directories(Eigen)

# Create Ray Tracer Executable
add_executable(raytracer main.cpp Camera.cpp Camera.h DReader.cpp DReader.h LightSource.cpp LightSource.h Material.cpp Material.h Object.cpp Object.h PWriter.cpp PWriter.h Ray.cpp Ray.h Remap.cpp Remap.h Sphere.cpp Sphere.h)

# Create a tar file every time
add_custom_target(mrosoff.tar ALL COMMAND tar cf mrosoff.tar *.cpp *.h README.md CMakeLists.txt Eigen)
