// tpetra_fixed_hash_table_openmp_symbols.cpp

#include <Kokkos_Core.hpp>

#include <Tpetra_Details_FixedHashTable_decl.hpp>
#include <Tpetra_Details_FixedHashTable_def.hpp>

namespace Tpetra::Details
{
  template class FixedHashTable<
      int,
      int,
      Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>>;
}
