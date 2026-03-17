cmake_minimum_required(VERSION 3.18)
project(PragmaticFS LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" HAVE_MARCH_NATIVE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories("${LW_INCLUDE_DIR}")
link_libraries(${CMAKE_DL_LIBS})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

add_library(pragmatic_fvm_fs_1st_1d SHARED pragmatic_fvm_fs.cpp)