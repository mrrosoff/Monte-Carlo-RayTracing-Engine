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
add_executable(raytracer main.cpp 
RayTracingFiles/RayTracing/RayTracer.cpp RayTracingFiles/RayTracing/RayTracer.h
RayTracingFiles/RayTracing/Camera.cpp RayTracingFiles/RayTracing/Camera.h
RayTracingFiles/RayTracing/Ray.cpp RayTracingFiles/RayTracing/Ray.h

RayTracingFiles/FileInOut/DReader.cpp RayTracingFiles/FileInOut/DReader.h
RayTracingFiles/FileInOut/PWriter.cpp RayTracingFiles/FileInOut/PWriter.h

RayTracingFiles/SceneItems/LightSource.cpp RayTracingFiles/SceneItems/LightSource.h
RayTracingFiles/SceneItems/Material.cpp RayTracingFiles/SceneItems/Material.h
RayTracingFiles/SceneItems/Sphere.cpp RayTracingFiles/SceneItems/Sphere.h
RayTracingFiles/SceneItems/SceneItem.cpp RayTracingFiles/SceneItems/SceneItem.h

RayTracingFiles/SceneItems/Models/Object.cpp RayTracingFiles/SceneItems/Models/Object.h
RayTracingFiles/SceneItems/Models/Remap.cpp RayTracingFiles/SceneItems/Models/Remap.h
RayTracingFiles/SceneItems/Models/Vertex.cpp RayTracingFiles/SceneItems/Models/Vertex.h
RayTracingFiles/SceneItems/Models/Face.cpp RayTracingFiles/SceneItems/Models/Face.h)

# Create a tar file every time
add_custom_target(mrosoff.tar ALL COMMAND tar cf mrosoff.tar ExampleDriverFiles main.cpp RayTracingFiles README.md CMakeLists.txt Eigen)
