cmake_minimum_required(VERSION 3.7.1)

file(GLOB C_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(ply_io SHARED ${C_FILES})

