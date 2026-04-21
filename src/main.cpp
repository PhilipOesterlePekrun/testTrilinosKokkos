#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <chrono>
#include <iostream>

#include "MPI_TpetraSerial_KokkosCuda.h"

int main(int argc, char* argv[])
{
  const auto startTime = std::chrono::steady_clock::now();
  
  int rank = 0, size = 1;
  {
    MPI_Init(&argc, &argv);
    struct CleanUpMPI
    {
      ~CleanUpMPI() { MPI_Finalize(); }
    } cleanup_mpi;

    Kokkos::ScopeGuard kokkos_guard(argc, argv);

    MPI_TpetraSerial_KokkosCuda(argc, argv);
  
}
  
  const double t =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();

  if (!rank) {
    std::cout << "Total wall time: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n";
  }

  return 0;
}
