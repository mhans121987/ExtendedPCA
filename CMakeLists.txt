cmake_minimum_required (VERSION 2.8.3)
project(pcastar)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall ${CMAKE_CXX_FLAGS}")

include_directories(src)

add_executable(example
  src/main.cpp)

add_library(pcastar
  src/pca_star.cpp)

target_link_libraries(example pcastar)

add_subdirectory(test)
