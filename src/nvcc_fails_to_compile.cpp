#include "nvcc_fails_to_compile.h"

#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>
#include <Teuchos_RCP.hpp>

#include <array>
#include <cstddef>
#include <iostream>

/*__host__ (tried; doesn't work)*/
template <std::size_t... n>
consteval auto breaks_nvcc()
{
  constexpr std::array shape = {n...};

  if constexpr (sizeof...(n) == 2)
  {
    std::array<std::size_t, shape[0] * shape[1]> out{};
    for (std::size_t j = 0; j < shape[1]; ++j)
      for (std::size_t i = 0; i < shape[0]; ++i)
        out[i + j * shape[0]] = i + 10 * j;
    return out;
  }
  else
  {
    std::array left = breaks_nvcc<shape[0], shape[1]>();
    std::array right = breaks_nvcc<shape[2], shape[3]>();

    std::array<std::size_t, left.size() * right.size()> out{};
    for (std::size_t j = 0; j < right.size(); ++j)
      for (std::size_t i = 0; i < left.size(); ++i)
        out[i + j * left.size()] = left[i] + right[j];
    return out;
  }
}

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
    
    ////////////////////////////////////////////
    {      
      using map_type = Tpetra::Map<>;
      using vec_type = Tpetra::Vector<>;
      
      using node_type = typename vec_type::node_type;
      using device_type = typename vec_type::device_type;
      using execution_space = typename vec_type::execution_space;
      using memory_space = typename device_type::memory_space;

      if (!rank) {
        std::cout << "-- Tpetra type information --\n";
        std::cout << "vec_type::node_type        = " << typeid(node_type).name() << '\n';
        std::cout << "vec_type::device_type      = " << typeid(device_type).name() << '\n';
        std::cout << "vec_type::execution_space  = " << typeid(execution_space).name() << '\n';
        std::cout << "vec_type::memory_space     = " << typeid(memory_space).name() << '\n';
        std::cout << '\n';
      }

      auto comm = Tpetra::getDefaultComm();
      auto map = Teuchos::rcp(new map_type(2, 0, comm));
      vec_type x(map);
      x.putScalar(1.0);
      
      constexpr auto trigger = breaks_nvcc<3, 3, 3, 3>();

      if (comm->getRank() == 0)
        std::cout << "trigger[0] = " << trigger[0]
                  << ", norm1 = " << x.norm1() << '\n';
    }
  }
  
  const double t =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();

  if (!rank) {
    std::cout << "Total wall time: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n";
  }

  return 0;
}
