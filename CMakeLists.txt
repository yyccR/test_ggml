cmake_minimum_required(VERSION 3.17)
project(test_ggml)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

include_directories(ggml-1.4.2)
include_directories(whisper)
include_directories(drwav)
set(whisper_src ${CMAKE_SOURCE_DIR}/whisper/whisper.cpp)

aux_source_directory(${CMAKE_SOURCE_DIR}/ggml-1.4.2 ggml_src)
add_library(ggml SHARED ${ggml_src} ${whisper_src})


add_executable(test_ggml main.cpp)
target_link_libraries(test_ggml ggml)