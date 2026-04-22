cmake_minimum_required(VERSION 3.28)
#project(testTrilinosKokkos VERSION 0.1.0)

project(testTrilinosKokkos LANGUAGES CXX)# CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)





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




















set(CMAKE_PREFIX_PATH ${TRILINOS_PATH} ${CMAKE_PREFIX_PATH})
find_package(Trilinos REQUIRED COMPONENTS Kokkos Tpetra)

find_package(mirco_lib CONFIG REQUIRED HINTS ${MIRCO_PATH})





#find_package(Kokkos REQUIRED COMPONENTS separable_compilation)
#enable_language(${Kokkos_COMPILE_LANGUAGE})



add_library(use_ryml
  src/use_ryml.cpp
  )
target_link_libraries(use_ryml PUBLIC ryml)



add_library(MPI_TpetraSerial_KokkosCuda
  src/MPI_TpetraSerial_KokkosCuda.cpp
  )
target_link_libraries(MPI_TpetraSerial_KokkosCuda PRIVATE mirco::mirco_lib PUBLIC Trilinos::all_selected_libs use_ryml)
#kokkos_compilation(TARGET MPI_TpetraSerial_KokkosCuda)

#mark_target_for_kokkos_backend(MPI_TpetraSerial_KokkosCuda)


add_executable(exe src/main.cpp)
target_link_libraries(exe PRIVATE MPI_TpetraSerial_KokkosCuda)


find_package(Tpetra REQUIRED)
set_source_files_properties(src/nvcc_fails_to_compile.cpp PROPERTIES LANGUAGE CXX)
add_executable(exe_nvcc_fails src/nvcc_fails_to_compile.cpp)
target_link_libraries(exe_nvcc_fails PRIVATE Tpetra::all_libs)#Trilinos::all_selected_libs)#Tpetra)

get_source_file_property(_lang
  src/nvcc_fails_to_compile.cpp
  TARGET_DIRECTORY exe_nvcc_fails
  LANGUAGE
)
message(STATUS "LANG = '${_lang}'")
message(STATUS "CMAKE_CXX_COMPILER = '${CMAKE_CXX_COMPILER}'")
