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
add_executable(raytracer main.cpp RayTracingFiles/RayTracer.cpp RayTracingFiles/RayTracer.h RayTracingFiles/Camera.cpp RayTracingFiles/Camera.h RayTracingFiles/DReader.cpp RayTracingFiles/DReader.h RayTracingFiles/LightSource.cpp RayTracingFiles/LightSource.h RayTracingFiles/Material.cpp RayTracingFiles/Material.h RayTracingFiles/Object.cpp RayTracingFiles/Object.h RayTracingFiles/PWriter.cpp RayTracingFiles/PWriter.h RayTracingFiles/Ray.cpp RayTracingFiles/Ray.h RayTracingFiles/Remap.cpp RayTracingFiles/Remap.h RayTracingFiles/Sphere.cpp RayTracingFiles/Sphere.h)

# Create a tar file every time
add_custom_target(mrosoff.tar ALL COMMAND tar cf mrosoff.tar *.cpp *.h README.md CMakeLists.txt Eigen)
