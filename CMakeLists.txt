cmake_minimum_required(VERSION 3.28)
#project(testTrilinosKokkos VERSION 0.1.0)

project(testTrilinosKokkos LANGUAGES CXX)# CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_PREFIX_PATH ${TRILINOS_PATH} ${CMAKE_PREFIX_PATH})
find_package(Trilinos REQUIRED COMPONENTS Kokkos Tpetra)

find_package(mirco_lib CONFIG REQUIRED HINTS ${MIRCO_PATH})



# DISABLE ALL WARNINGS
if(true)
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

  if(Kokkos_ENABLE_CUDA)
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CUDA>:-w>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
    )
  endif()
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

add_library(use_ryml
  src/use_ryml.cpp
  )
target_link_libraries(use_ryml PUBLIC ryml)



add_library(MPI_TpetraSerial_KokkosCuda
  src/MPI_TpetraSerial_KokkosCuda.cpp
  )
target_link_libraries(MPI_TpetraSerial_KokkosCuda PRIVATE mirco::mirco_lib PUBLIC Trilinos::all_selected_libs use_ryml)



if(false)#Kokkos_ENABLE_CUDA)
  add_executable(exe src/main.cu)
  
  target_include_directories(exe PRIVATE
    /opt/openmpi/4.1.8/gcc/include
    /opt/openmpi/4.1.8/gcc/include/openmpi
  )
  target_link_directories(exe PRIVATE
    /opt/openmpi/4.1.8/gcc/lib
  )
  target_link_libraries(exe PRIVATE mpi)
  
  
  if(false)
  set(MPI_ASSUME_NO_BUILTIN_MPI TRUE)
  set(MPI_CXX_COMPILER /opt/openmpi/4.1.8/gcc/bin/mpicxx)
  find_package(MPI REQUIRED COMPONENTS CXX)

  target_include_directories(exe PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  target_compile_options(exe PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
  target_link_options(exe PRIVATE ${MPI_CXX_LINK_FLAGS})
  target_link_libraries(exe PRIVATE ${MPI_CXX_LIBRARIES})
  message(STATUS "MPI_CXX_FOUND=${MPI_CXX_FOUND}")
  message(STATUS "MPI_CXX_COMPILER=${MPI_CXX_COMPILER}")
  endif()
else()
  add_executable(exe src/main.cpp)
endif()
target_link_libraries(exe PRIVATE MPI_TpetraSerial_KokkosCuda)

add_executable(exe_nvcc_fails src/nvcc_fails_to_compile.cpp)
target_link_libraries(exe_nvcc_fails PRIVATE Trilinos::all_selected_libs)#Tpetra)

