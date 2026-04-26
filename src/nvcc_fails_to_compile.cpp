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
  Tpetra::ScopeGuard scope(&argc, &argv);
  {
    const auto startTime = std::chrono::steady_clock::now();
    
    auto comm = Tpetra::getDefaultComm();
    int rank = comm->getRank();
    int size = comm->getSize();
  
    using LO = int;
    using GO = int;
    using map_type = Tpetra::Map<LO, GO>;
    using vec_type = Tpetra::Vector<double, LO, GO>;
      
      
          
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
      
      #if defined(__clang__) && defined(__CUDA__)
      #pragma message("Using Clang CUDA")
      #endif
    }

    const LO local_num_rows = 200000000;
    auto map = Teuchos::rcp(
      new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
        local_num_rows, 0, comm));

    vec_type x(map);

    x.putScalar(1.0);
    x.scale(2.0);
    auto xNorm = x.norm2();
    
    constexpr auto trigger = breaks_nvcc<3, 3, 3, 3>();
    
    if (!rank) {
      std::cout << "nvcc_fails_to_compile work:\n";
      std::cout << "x.norm2() = " << xNorm << "\n";
      std::cout << "trigger[0] = "<< trigger[0] << "\n\n";
    }
    
    const double t =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
    if (!rank) {
      std::cout << "Total wall time: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n";
    }
  }
  
  return 0;
}
