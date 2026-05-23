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
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  if(!rank) {
    std::cout << "-- MPI information --\n";
    std::cout << "size: " << size << "\n\n";
  }

  Kokkos::ScopeGuard kokkos_guard(argc, argv);
  
  
  // Tpetra (with host NO because Tpetra_INST_CUDA=OFF)
  {
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
    }

    auto comm = Teuchos::rcp(
    new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(MPI_COMM_WORLD)));

    const LO local_num_rows = 200000000;
    auto map = Teuchos::rcp(
      new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
        local_num_rows, 0, comm));

    vec_type x(map);

    x.putScalar(1.0);
    x.scale(2.0);
    auto xNorm = x.norm2();
    if (!rank) {
      std::cout << "Tpetra work: ";
      std::cout << "x.norm2() = " << xNorm << "\n\n";
    }
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  if(!rank)
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
    
    if(!rank) {
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
        
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int node_rank, node_size;
    MPI_Comm_rank(comm_node, &node_rank);
    MPI_Comm_size(comm_node, &node_size);
    
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int proc_name_len = 0;
    MPI_Get_processor_name(proc_name, &proc_name_len);
    
    
    for (int owner = 0; owner < node_size; ++owner) {
      if (node_rank == owner) {
        const int n = 400000000;
        ViewVector_d X_d("X", n);
        Kokkos::deep_copy(X_d, 0);
        Kokkos::parallel_for(
          n, KOKKOS_LAMBDA(const int i) {
            X_d(i) = 10*rank;
          });
        
        auto X_h = Kokkos::create_mirror_view_and_copy(ExecSpace_DefaultHost_t(), X_d);
        
        std::cout << "Kokkos work (node_rank="<<node_rank<<"; rank="<<rank<<"): "
          << "X_h(n/2)="<<X_h(n/2)<<"\n";
        std::cout << "\tTIME: "<<std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n\n";
      }
      
      MPI_Barrier(comm_node);
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (!rank) {
    std::cout << "MPI_TpetraSerial_Kokkos_d__SERIALIZE total wall time: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n";
  }

  return 0;
}
