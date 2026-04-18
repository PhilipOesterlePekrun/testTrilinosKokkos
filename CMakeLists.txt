cmake_minimum_required(VERSION 3.28)
project(testTrilinosKokkos VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH ${TRILINOS_PATH} ${CMAKE_PREFIX_PATH})
find_package(Trilinos REQUIRED COMPONENTS Kokkos Tpetra)

find_package(mirco_lib CONFIG HINTS ${MIRCO_PATH})

if(true)
add_executable(exe src/main.cpp)
target_link_libraries(exe PRIVATE Trilinos::all_selected_libs mirco::mirco_lib)
else()
add_executable(exe2 src/main2.cpp)
target_link_libraries(exe2 PRIVATE Trilinos::all_selected_libs)
endif()
