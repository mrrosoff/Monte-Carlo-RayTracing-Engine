# Created by Max Rosoff on 10/19/2019.

cmake_minimum_required(VERSION 3.10)
project(Graphics)

set(CMAKE_CXX_STANDARD 14)

# Compile Flags
add_compile_options(-Wall -O3)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Include Matrix Library
include_directories(Eigen)

# Create the executable raytracer from the source files
add_executable(raytracer main.cpp DReader.cpp DReader.h PWriter.cpp PWriter.h
Camera.cpp Camera.h Ray.cpp Ray.h LightSource.cpp LightSource.h Material.cpp Material.h
Remap.cpp Remap.h Object.cpp Object.h Sphere.cpp Sphere.h)

# Create a tar file every time:
add_custom_target(mrosoffA3.tar ALL COMMAND tar cf mrosoffA3.tar *.cpp *.h ../drivers/driver02.txt README.md Eigen CMakeLists.txt)
