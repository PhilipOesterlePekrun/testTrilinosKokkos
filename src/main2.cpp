#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_RCP.hpp>

#include <iostream>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  struct CleanUpMPI
  {
    ~CleanUpMPI() { MPI_Finalize(); }
  } cleanup_mpi;

  Kokkos::ScopeGuard kokkos_guard(argc, argv);

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Independent Kokkos work
  double local_micro_value = 0.0;
  {
    Kokkos::View<double*> micro("micro", 8);

    Kokkos::parallel_for(
      "fill_micro",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 8),
      KOKKOS_LAMBDA(const int i) {
        micro(i) = 0.1 * (i + 1) * (rank + 1);
      });

    Kokkos::parallel_reduce(
      "sum_micro",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 8),
      KOKKOS_LAMBDA(const int i, double& sum) {
        sum += micro(i);
      },
      local_micro_value);

    Kokkos::fence();
  }

  double coupled_scalar = 0.0;
  MPI_Allreduce(&local_micro_value, &coupled_scalar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  {
    using LO = int;
    using GO = int;
    using map_type = Tpetra::Map<LO, GO>;
    using vec_type = Tpetra::Vector<double, LO, GO>;

    auto comm = Teuchos::rcp(
      new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(MPI_COMM_WORLD)));

    const LO local_num_rows = 5;
    const Tpetra::global_size_t global_num_rows =
      static_cast<Tpetra::global_size_t>(local_num_rows) * size;

    auto map = Teuchos::rcp(new map_type(global_num_rows, local_num_rows, 0, comm));
    vec_type x(map);

    x.putScalar(coupled_scalar);

    auto x_host = x.getLocalViewHost(Tpetra::Access::ReadOnly);

    double local_tpetra_sum = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(x.getLocalLength()); ++i) {
      local_tpetra_sum += x_host(i, 0);
    }

    double global_tpetra_sum = 0.0;
    Teuchos::reduceAll(
      *comm,
      Teuchos::REDUCE_SUM,
      1,
      &local_tpetra_sum,
      &global_tpetra_sum);

    if (rank == 0) {
      std::cout << "DefaultExecutionSpace = "
                << Kokkos::DefaultExecutionSpace::name() << '\n';
      std::cout << "coupled_scalar = " << coupled_scalar << '\n';
      std::cout << "global_tpetra_sum = " << global_tpetra_sum << '\n';
    }
  }

  return 0;
}