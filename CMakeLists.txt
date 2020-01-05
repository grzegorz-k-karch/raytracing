cmake_minimum_required(VERSION 3.7.1)

project(raytracing LANGUAGES CXX CUDA)

file(GLOB_RECURSE CU_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)

add_executable(raytracing-cuda ${CU_FILES})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set_target_properties(raytracing-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(raytracing-cuda PUBLIC cxx_std_11)

target_link_libraries(raytracing-cuda boost_program_options)
