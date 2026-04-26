cmake_minimum_required(VERSION 3.28)
#project(testTrilinosKokkos VERSION 0.1.0)

project(testTrilinosKokkos LANGUAGES CXX)# CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


message(STATUS "CMAKE_CXX_COMPILER      = ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER_ID   = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER = ${CMAKE_CXX_COMPILER_LAUNCHER}")
message(STATUS "CMAKE_CXX_FLAGS         = ${CMAKE_CXX_FLAGS}")


set(CMAKE_PREFIX_PATH ${TRILINOS_PATH} ${CMAKE_PREFIX_PATH})
find_package(Trilinos REQUIRED COMPONENTS Kokkos Tpetra)

set(CMAKE_CXX_COMPILER_LAUNCHER "" CACHE STRING "" FORCE)
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "")

# DISABLE ALL WARNINGS
if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C>:/w>
    $<$<COMPILE_LANGUAGE:CXX>:/w>
  )
else()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C>:-w>
    $<$<COMPILE_LANGUAGE:CXX>:-w>
  )
endif()

if(CMAKE_CUDA_COMPILER)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
  )
endif()

message(STATUS "Fetching rapidyaml...")
include(FetchContent)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE) # Suppress developer warnings from ryml=
FetchContent_Declare(rapidyaml
  GIT_REPOSITORY https://github.com/biojppm/rapidyaml.git
  GIT_TAG 47ec2fa184209687c20fd5bc05621e1cb1200311
)
FetchContent_MakeAvailable(rapidyaml)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS OFF CACHE BOOL "" FORCE)

find_package(MPI REQUIRED COMPONENTS CXX)

add_library(MPI_TpetraSerial_Kokkos_d
  src/MPI_TpetraSerial_Kokkos_d.cpp
  )
target_link_libraries(MPI_TpetraSerial_Kokkos_d PUBLIC Trilinos::all_selected_libs ryml MPI::MPI_CXX)


add_executable(exe src/main.cpp)
target_link_libraries(exe PRIVATE MPI_TpetraSerial_Kokkos_d)

find_package(Tpetra REQUIRED)
add_executable(exe_nvcc_fails src/nvcc_fails_to_compile.cpp)

set_property(TARGET exe_nvcc_fails PROPERTY CXX_COMPILER_LAUNCHER "")

target_compile_options(exe_nvcc_fails PRIVATE
  -x cuda
  --cuda-gpu-arch=sm_90
  -Wno-unknown-cuda-version
)

target_link_libraries(exe_nvcc_fails PRIVATE Trilinos::all_selected_libs MPI::MPI_CXX) # OR Trilinos::all_selected_libs, really does not matter
