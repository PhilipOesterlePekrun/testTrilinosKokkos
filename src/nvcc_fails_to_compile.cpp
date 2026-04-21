#include "nvcc_fails_to_compile.h"

#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>
#include <Teuchos_RCP.hpp>

#include <array>
#include <cstddef>
#include <iostream>

template <std::size_t... n>
consteval auto weird_index_map()
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
    std::array left = weird_index_map<shape[0], shape[1]>();
    std::array right = weird_index_map<shape[2], shape[3]>();

    std::array<std::size_t, left.size() * right.size()> out{};
    for (std::size_t j = 0; j < right.size(); ++j)
      for (std::size_t i = 0; i < left.size(); ++i)
        out[i + j * left.size()] = left[i] + right[j];
    return out;
  }
}

constexpr auto trigger = weird_index_map<3, 3, 3, 3>();

int main(int argc, char* argv[])
{
  Tpetra::ScopeGuard scope(&argc, &argv);
  {
    using map_type = Tpetra::Map<>;
    using vec_type = Tpetra::Vector<>;

    auto comm = Tpetra::getDefaultComm();
    auto map = Teuchos::rcp(new map_type(5, 0, comm));
    vec_type x(map);
    x.putScalar(1.0);

    if (comm->getRank() == 0)
      std::cout << "trigger[0] = " << trigger[0]
                << ", norm1 = " << x.norm1() << '\n';
  }
  return 0;
}
