cmake_minimum_required(VERSION 3.28)
#project(testTrilinosKokkos VERSION 0.1.0)

project(testTrilinosKokkos LANGUAGES CXX)# CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH ${TRILINOS_PATH} ${CMAKE_PREFIX_PATH})
find_package(Trilinos REQUIRED COMPONENTS Kokkos Tpetra)

find_package(mirco_lib CONFIG HINTS ${MIRCO_PATH})



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

  if(CMAKE_CUDA_COMPILER)
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CUDA>:-w>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
    )
  endif()
endif()




if(Kokkos_ENABLE_CUDA)
  message("============ CRIT { ============")
  message("Kokkos_ENABLE_CUDA=true; enabling language CUDA")
  message("============ CRIT } ============")
  
  enable_language(CUDA)
endif()

message(STATUS "Fetching rapidyaml...")
include(FetchContent)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE) # Suppress developer warnings from ryml

FetchContent_Declare(rapidyaml
  GIT_REPOSITORY https://github.com/biojppm/rapidyaml.git
  GIT_TAG 47ec2fa184209687c20fd5bc05621e1cb1200311
)
FetchContent_MakeAvailable(rapidyaml)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS OFF CACHE BOOL "" FORCE)





add_library(inTarget
  src/in.cpp
  )
target_link_libraries(inTarget PUBLIC ryml)



if(Kokkos_ENABLE_CUDA)
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

target_link_libraries(exe PRIVATE Trilinos::all_selected_libs mirco::mirco_lib inTarget)
