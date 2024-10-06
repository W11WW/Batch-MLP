cmake_minimum_required(VERSION 3.20)
project(AiLearning)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_BUILD_TYPE Debug)


add_executable(AiLearning main.cpp stb_image.h stb.cpp Vector.h Math.h Weight.h Layer.h BatchMLP.h BatchMLP.cpp)
