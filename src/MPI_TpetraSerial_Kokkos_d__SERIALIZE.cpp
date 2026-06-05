// This file uses raw MPI, Tpetra with serial/host backend (because Tpetra_INST_CUDA=off), and raw Kokkos with cuda backend (built) provided by MIRCO

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_RCP.hpp>

#include <chrono>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  struct CleanUpMPI
  {
    ~CleanUpMPI() { MPI_Finalize(); }
  } cleanup_mpi;
  
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  if(!world_rank) {
    std::cout << "-- MPI information --\n";
    std::cout << "world_size: " << world_size << "\n\n";
  }

  Kokkos::ScopeGuard kokkos_guard(argc, argv);
  
  
  // Tpetra (with host NO because Tpetra_INST_CUDA=OFF)
  {
    const auto startTime = std::chrono::steady_clock::now();
    
    using LO = int;
    using GO = int;
    using map_type = Tpetra::Map<LO, GO>;
    using vec_type = Tpetra::Vector<double, LO, GO>;
          
    using node_type = typename vec_type::node_type;
    using device_type = typename vec_type::device_type;
    using execution_space = typename vec_type::execution_space;
    using memory_space = typename device_type::memory_space;

    if (!world_rank) {
      std::cout << "-- Tpetra type information --\n";
      std::cout << "vec_type::node_type        = " << typeid(node_type).name() << '\n';
      std::cout << "vec_type::device_type      = " << typeid(device_type).name() << '\n';
      std::cout << "vec_type::execution_space  = " << typeid(execution_space).name() << '\n';
      std::cout << "vec_type::memory_space     = " << typeid(memory_space).name() << '\n';
      std::cout << '\n';
    }

    auto comm = Teuchos::rcp(
    new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(MPI_COMM_WORLD)));

    const LO local_num_rows = 20000000;
    auto map = Teuchos::rcp(
      new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
        local_num_rows, 0, comm));

    vec_type x(map);

    x.putScalar(1.0);
    x.scale(2.0);
    auto xNorm = x.norm2();
    if (!world_rank) {
      std::cout << "Tpetra work: ";
      std::cout << "x.norm2() = " << xNorm << "\n\n";
    }
    
    
    
    MPI_Barrier(MPI_COMM_WORLD);
  
    if (!world_rank) {
      std::cout << "Tpetra time: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  if(!world_rank)
    std::cout<<"\n\n\n\n\n";
  
  const auto startTime = std::chrono::steady_clock::now();
  
  // Pure Kokkos
  {
    using ExecSpace_DefaultHost_t = Kokkos::DefaultHostExecutionSpace;
    using ExecSpace_Default_t = Kokkos::DefaultExecutionSpace;
    using MemorySpace_Host_t = Kokkos::HostSpace;
    using MemorySpace_ofDefaultExec_t = ExecSpace_Default_t::memory_space;
    using Device_Host_t = Kokkos::Device<ExecSpace_DefaultHost_t, MemorySpace_Host_t>;
    using Device_Default_t = Kokkos::Device<ExecSpace_Default_t, MemorySpace_ofDefaultExec_t>;

    using ViewVector_d = Kokkos::View<double*, Kokkos::LayoutLeft, Device_Default_t>;
    using ViewMatrix_d = Kokkos::View<double**, Kokkos::LayoutLeft, Device_Default_t>;
    
    if(!world_rank) {
      std::cout << "-- Kokkos information --\n";
      std::cout << "Threads in use: " << ExecSpace_Default_t().concurrency() << "\n";
      std::cout << "Default execution space: " << typeid(ExecSpace_Default_t).name() << "\n";
      std::cout << "Default host execution space: " << typeid(ExecSpace_DefaultHost_t).name() << "\n";
      std::cout << "Default memory space: " << typeid(MemorySpace_ofDefaultExec_t).name() << "\n";
      std::cout << "Default host memory space: " << typeid(MemorySpace_Host_t).name() << "\n";
      std::cout << "Num devices = " << Kokkos::num_devices() << "\n";
      std::cout << "\n";      
    }
    
    
    MPI_Comm comm_node = MPI_COMM_NULL;
    MPI_Comm_split_type(
        MPI_COMM_WORLD,
        MPI_COMM_TYPE_SHARED,
        0,
        MPI_INFO_NULL,
        &comm_node);

    int node_rank, node_size;
    MPI_Comm_rank(comm_node, &node_rank);
    MPI_Comm_size(comm_node, &node_size);
    
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int proc_name_len = 0;
    MPI_Get_processor_name(proc_name, &proc_name_len);
    
    
    for (int owner = 0; owner < node_size; ++owner) {
      if (node_rank == owner) {
        /*
        const int n = 80000000;
        ViewVector_d X_d("X", n);
        Kokkos::deep_copy(X_d, 0);
        Kokkos::parallel_for(
          n, KOKKOS_LAMBDA(const int i) {
            X_d(i) = 10*world_rank;
          });
        
        auto X_h = Kokkos::create_mirror_view_and_copy(ExecSpace_DefaultHost_t(), X_d);
        
        std::cout << "Kokkos work (node_rank="<<node_rank<<"; world_rank="<<world_rank<<"): "
          << "X_h(n/2)="<<X_h(n/2)<<"\n";*/
          
        
        const int n = 8000000;
        const int num_iters = 20;

        ViewVector_d X_d("X", n);
        ViewVector_d Y_d("Y", n);
        ViewVector_d Z_d("Z", n);

        Kokkos::parallel_for(
          "init",
          n, KOKKOS_LAMBDA(const int i) {
            const double a = static_cast<double>((i % 97) + 1);
            X_d(i) = static_cast<double>(node_rank) + 0.001 * a;
            Y_d(i) = 0.002 * a;
            Z_d(i) = 0.0;
          });

        for (int iter = 0; iter < num_iters; ++iter) {
          Kokkos::parallel_for(
            "stencil_work",
            n - 2, KOKKOS_LAMBDA(const int j) {
              const int i = j + 1;

              const double xm = X_d(i - 1);
              const double xi = X_d(i);
              const double xp = X_d(i + 1);

              const double lap = xm - 2.0 * xi + xp;

              double v = xi + 0.05 * lap + 0.01 * Y_d(i);

              for (int k = 0; k < 16; ++k) {
                v = 0.999 * v
                  + 0.0005 * lap
                  + 0.0001 * Y_d(i)
                  + 0.000001 * static_cast<double>(k + 1);
              }

              Z_d(i) = v;
            });

          Kokkos::parallel_for(
            "boundary",
            2, KOKKOS_LAMBDA(const int i) {
              if (i == 0) {
                Z_d(0) = X_d(0);
              } else {
                Z_d(n - 1) = X_d(n - 1);
              }
            });

          Kokkos::parallel_for(
            "update",
            n, KOKKOS_LAMBDA(const int i) {
              Y_d(i) = 0.95 * Y_d(i) + 0.05 * Z_d(i);
              X_d(i) = Z_d(i);
            });
        }

        double checksum = 0.0;
        Kokkos::parallel_reduce(
          "checksum",
          n, KOKKOS_LAMBDA(const int i, double& local_sum) {
            local_sum += X_d(i) * X_d(i);
          },
          checksum);

        std::cout << "Kokkos work (node_rank="<<node_rank<<"; world_rank="<<world_rank<<"): "
          << "checksum="<<std::sqrt(checksum)<<"\n";
          
          
          
        std::cout << "\tTIME: "<<std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n\n";
      }
      
      MPI_Barrier(comm_node);
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (!world_rank) {
    std::cout << "MPI_TpetraSerial_Kokkos_d__SERIALIZE total wall time: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n";
  }

  return 0;
}
