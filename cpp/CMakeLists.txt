cmake_minimum_required(VERSION 3.16)
project(md_optimiser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ortools REQUIRED)

add_library(optimiser SHARED optimiser.cpp)
target_link_libraries(optimiser ortools::ortools)
