cmake_minimum_required(VERSION 3.20)
project(AiLearning)

set(CMAKE_CXX_STANDARD 20)

add_executable(AiLearning main.cpp stb_image.h stb.cpp Vector.h Math.h Options.cpp TSA.cpp Weight.h Layer.h BatchMLP.h BatchMLP.cpp Activation.h Activation.cpp)
