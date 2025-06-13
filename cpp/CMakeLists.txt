cmake_minimum_required(VERSION 3.16)
project(md_optimiser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(optimiser SHARED optimiser.cpp)

find_package(ortools REQUIRED)
target_link_libraries(optimiser PRIVATE ortools::ortools)

if(APPLE)
  set_target_properties(optimiser PROPERTIES SUFFIX ".dylib")
  # Duplicate .dylib as .so so Go's embed doesn't fail
  add_custom_command(TARGET optimiser POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      $<TARGET_FILE:optimiser>
      ${CMAKE_CURRENT_BINARY_DIR}/liboptimiser.so
  )
endif()
