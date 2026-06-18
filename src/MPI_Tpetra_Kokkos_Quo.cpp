#define USE_QUO 1

#include <mpi.h>

#if USE_QUO
#include <quo.h>
#endif

#include <Kokkos_Core.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_RCP.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <typeinfo>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  {
    Kokkos::ScopeGuard kokkos_guard(argc, argv);

    int node_rank = 0;
    int node_size = 0;

#if USE_QUO
    QUO_context quo = nullptr;

    if (QUO_SUCCESS != QUO_create(&quo, MPI_COMM_WORLD)) {
      std::cerr << "QUO_create failed on rank " << world_rank << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    QUO_id(quo, &node_rank);
    QUO_nqids(quo, &node_size);
#else
    MPI_Comm comm_node = MPI_COMM_NULL;

    MPI_Comm_split_type(
        MPI_COMM_WORLD,
        MPI_COMM_TYPE_SHARED,
        0,
        MPI_INFO_NULL,
        &comm_node);

    MPI_Comm_rank(comm_node, &node_rank);
    MPI_Comm_size(comm_node, &node_size);
#endif

    if (world_rank == 0) {
      std::cout << "USE_QUO = " << USE_QUO << "\n";
      std::cout << "world_size = " << world_size << "\n";
      std::cout << "node_size = " << node_size << "\n";
      std::cout << "Kokkos concurrency = "
                << Kokkos::DefaultExecutionSpace().concurrency() << "\n";
      std::cout << "Kokkos default execution space = "
                << typeid(Kokkos::DefaultExecutionSpace).name() << "\n\n";
    }

    // -------------------------------------------------------------------------
    // Region 1: Tpetra with default template parameters
    // -------------------------------------------------------------------------
    {
      MPI_Barrier(MPI_COMM_WORLD);

      const auto start_time = std::chrono::steady_clock::now();

      using LO = int;
      using GO = int;

      using map_type = Tpetra::Map<LO, GO>;
      using vec_type = Tpetra::Vector<double, LO, GO>;

      if (world_rank == 0) {
        std::cout << "-- Tpetra region --\n";
        std::cout << "vec_type::node_type       = "
                  << typeid(typename vec_type::node_type).name() << "\n";
        std::cout << "vec_type::device_type     = "
                  << typeid(typename vec_type::device_type).name() << "\n";
        std::cout << "vec_type::execution_space = "
                  << typeid(typename vec_type::execution_space).name() << "\n";
      }

      auto comm = Teuchos::rcp(
          new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(MPI_COMM_WORLD)));

      const Tpetra::global_size_t global_num_rows = 200000000;

      auto map = Teuchos::rcp(new map_type(global_num_rows, 0, comm));

      vec_type x(map);
      x.putScalar(1.0);
      x.scale(2.0);

      const double norm = x.norm2();

      const double local_time =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - start_time)
              .count();

      double max_time = 0.0;
      MPI_Reduce(
          &local_time,
          &max_time,
          1,
          MPI_DOUBLE,
          MPI_MAX,
          0,
          MPI_COMM_WORLD);

      if (world_rank == 0) {
        std::cout << "Tpetra norm = " << norm << "\n";
        std::cout << "Tpetra max time = " << max_time << " s\n\n";
      }
    }

    // -------------------------------------------------------------------------
// Region 2: fixed number of distributed MIRCO-like problems
// -------------------------------------------------------------------------
{
  MPI_Barrier(MPI_COMM_WORLD);

  const int num_problems = 16;
  const int n = 200000000;
  const int num_iters = 20;

  if (world_rank == 0) {
    std::cout << "-- distributed serialized Kokkos problems --\n";
    std::cout << "num_problems = " << num_problems << "\n";
    std::cout << "n = " << n << "\n";
    std::cout << "num_iters = " << num_iters << "\n\n";
  }

  const auto section_start_time = std::chrono::steady_clock::now();

  int local_num_problems_done = 0;
  double local_active_time = 0.0;

  for (int problem_id = 0; problem_id < num_problems; ++problem_id) {
    const int owner = problem_id % node_size;

    if (node_rank == owner) {
#if USE_QUO
      char* before = nullptr;
      char* pushed = nullptr;
      char* after = nullptr;

      QUO_stringify_cbind(quo, &before);

      if (QUO_SUCCESS != QUO_bind_push(
              quo,
              QUO_BIND_PUSH_OBJ,
              QUO_OBJ_MACHINE,
              -1)) {
        std::cerr << "QUO_bind_push failed on rank " << world_rank << "\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
      }

      QUO_stringify_cbind(quo, &pushed);
#endif

      const auto problem_start_time = std::chrono::steady_clock::now();

      using exec_space = Kokkos::DefaultExecutionSpace;
      using device_type =
          Kokkos::Device<exec_space, typename exec_space::memory_space>;

      using view_type =
          Kokkos::View<double*, Kokkos::LayoutLeft, device_type>;

      view_type x("x", n);
      view_type y("y", n);

      Kokkos::parallel_for(
          "init",
          Kokkos::RangePolicy<exec_space>(0, n),
          KOKKOS_LAMBDA(const int i) {
            x(i) =
                1.0 +
                0.000001 * static_cast<double>((i + problem_id) % 97);
            y(i) = 0.0;
          });

      for (int iter = 0; iter < num_iters; ++iter) {
        Kokkos::parallel_for(
            "work",
            Kokkos::RangePolicy<exec_space>(1, n - 1),
            KOKKOS_LAMBDA(const int i) {
              y(i) = 0.25 * x(i - 1) + 0.5 * x(i) + 0.25 * x(i + 1);
            });

        Kokkos::parallel_for(
            "update",
            Kokkos::RangePolicy<exec_space>(1, n - 1),
            KOKKOS_LAMBDA(const int i) {
              x(i) = y(i);
            });
      }

      double sum = 0.0;

      Kokkos::parallel_reduce(
          "sum",
          Kokkos::RangePolicy<exec_space>(0, n),
          KOKKOS_LAMBDA(const int i, double& local_sum) {
            local_sum += x(i) * x(i);
          },
          sum);

      Kokkos::fence();

      const double problem_time =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - problem_start_time)
              .count();

      local_active_time += problem_time;
      ++local_num_problems_done;

#if USE_QUO
      if (QUO_SUCCESS != QUO_bind_pop(quo)) {
        std::cerr << "QUO_bind_pop failed on rank " << world_rank << "\n";
        MPI_Abort(MPI_COMM_WORLD, 3);
      }

      QUO_stringify_cbind(quo, &after);
#endif

      std::cout << "problem_id=" << problem_id
                << " world_rank=" << world_rank
                << " node_rank=" << node_rank
                << " time=" << problem_time
                << " checksum=" << std::sqrt(sum) << "\n";

#if USE_QUO
      std::cout << "  before: " << (before ? before : "null") << "\n";
      std::cout << "  pushed: " << (pushed ? pushed : "null") << "\n";
      std::cout << "  after:  " << (after ? after : "null") << "\n";

      std::free(before);
      std::free(pushed);
      std::free(after);
#endif

      std::cout << "\n";
    }

#if USE_QUO
    if (QUO_SUCCESS != QUO_barrier(quo)) {
      std::cerr << "QUO_barrier failed on rank " << world_rank << "\n";
      MPI_Abort(MPI_COMM_WORLD, 4);
    }
#else
    MPI_Barrier(comm_node);
#endif
  }

  const double section_time =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - section_start_time)
          .count();

  double max_section_time = 0.0;
  double sum_active_time = 0.0;
  int total_problems_done = 0;

  MPI_Reduce(
      &section_time,
      &max_section_time,
      1,
      MPI_DOUBLE,
      MPI_MAX,
      0,
      MPI_COMM_WORLD);

  MPI_Reduce(
      &local_active_time,
      &sum_active_time,
      1,
      MPI_DOUBLE,
      MPI_SUM,
      0,
      MPI_COMM_WORLD);

  MPI_Reduce(
      &local_num_problems_done,
      &total_problems_done,
      1,
      MPI_INT,
      MPI_SUM,
      0,
      MPI_COMM_WORLD);

  if (world_rank == 0) {
    std::cout << "total_problems_done = " << total_problems_done << "\n";
    std::cout << "sum active problem time = " << sum_active_time << " s\n";
    std::cout << "serialized section max time = "
              << max_section_time << " s\n";
  }
}

#if USE_QUO
    QUO_free(quo);
#else
    MPI_Comm_free(&comm_node);
#endif
  }

  MPI_Finalize();
  return 0;
}