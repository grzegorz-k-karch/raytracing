cmake_minimum_required(VERSION 3.7.1)

project(raytracing LANGUAGES CXX CUDA)

find_package(Boost 1.71 REQUIRED COMPONENTS log_setup log)

file(GLOB_RECURSE CU_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
file(GLOB CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(raytracing-cuda ${CU_FILES} ${CPP_FILES})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set_target_properties(raytracing-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(raytracing-cuda PUBLIC cxx_std_14)

target_include_directories(raytracing-cuda PRIVATE "/usr/local/cuda/include")

# nvidia sdk
target_include_directories(raytracing-cuda PRIVATE ${CMAKE_SOURCE_DIR}/src/external/)
# ply_io
target_include_directories(raytracing-cuda PRIVATE ${CMAKE_SOURCE_DIR}/src/external/ply_io/include)
target_link_directories(raytracing-cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/external/ply_io/build)

target_link_libraries(raytracing-cuda
  boost_program_options
  ${Boost_LIBRARIES}
  ply_io
)
