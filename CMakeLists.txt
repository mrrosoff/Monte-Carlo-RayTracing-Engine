cmake_minimum_required(VERSION 3.10)
project(Graphics)

set(CMAKE_CXX_STANDARD 14)

include_directories(../Eigen)
add_executable(Graphics main.cpp Remap.cpp Remap.h DReader.cpp DReader.h OReader.cpp OReader.h Writer.cpp Writer.h)
