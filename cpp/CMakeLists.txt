cmake_minimum_required(VERSION 3.16)
project(md_optimiser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(optimiser SHARED optimiser.cpp)

if(DEFINED ENV{ORTOOLS_HOME})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{ORTOOLS_HOME}")
endif()

find_package(ortools REQUIRED CONFIG)
target_link_libraries(optimiser PRIVATE ortools::ortools)

set_target_properties(optimiser PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# Apple fix for Go embed
if(APPLE)
  set_target_properties(optimiser PROPERTIES SUFFIX ".dylib")
  add_custom_command(TARGET optimiser POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      $<TARGET_FILE:optimiser>
      ${CMAKE_CURRENT_BINARY_DIR}/liboptimiser.so
  )
endif()
