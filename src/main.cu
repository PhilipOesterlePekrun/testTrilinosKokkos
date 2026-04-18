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

#include <mirco_evaluate.h>
#include <mirco_inputparameters.h>
#include <mirco_kokkostypes.h>
#include <mirco_topologyutilities.h>

#include "in.h"

int main(int argc, char* argv[])
{
  const auto startTime = std::chrono::steady_clock::now();
  auto betweenTime = std::chrono::steady_clock::now();
  
  int rank = 0, size = 1;
  {
    MPI_Init(&argc, &argv);
    struct CleanUpMPI
    {
      ~CleanUpMPI() { MPI_Finalize(); }
    } cleanup_mpi;

    Kokkos::ScopeGuard kokkos_guard(argc, argv);
    
    using namespace MIRCO;
    
    
    
    
    

    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(rank==0) {
      std::cout << "-- Kokkos information --\n";
      std::cout << "Threads in use: " << ExecSpace_Default_t().concurrency() << "\n";
      std::cout << "Default execution space: " << typeid(ExecSpace_Default_t).name() << "\n";
      std::cout << "Default host execution space: " << typeid(ExecSpace_DefaultHost_t).name() << "\n";
      std::cout << "Default memory space: " << typeid(MemorySpace_ofDefaultExec_t).name() << "\n";
      std::cout << "Default host memory space: " << typeid(MemorySpace_Host_t).name() << "\n";
      std::cout << "\n";
      
      std::cout << "num devices = " << Kokkos::num_devices() << '\n';
    }

    {


      const auto start = std::chrono::high_resolution_clock::now();

      InputParameters inputParams(1, 1, 0.3, 0.3, 1e-6, 5, 1000, 7, 5, 0.5, 100, false, true, false, 120);

      ViewVector_d meshgrid = CreateMeshgrid(inputParams.N, inputParams.grid_size);
      const double topologyMax = GetMax(inputParams.topology);

      // Main evaluation agorithm
      double meanPressure, effectiveContactAreaFraction;
      Evaluate(meanPressure, effectiveContactAreaFraction, inputParams, topologyMax, meshgrid);

      const auto finish = std::chrono::high_resolution_clock::now();

      /*std::cout << std::setprecision(16) << "Mean pressure is: " << meanPressure
                << "\nEffective contact area fraction is: " << effectiveContactAreaFraction
                << std::endl;*/

      //const double elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count();
      //std::cout << "Elapsed time is: " + std::to_string(elapsedTime) + "s" << std::endl;
    }

    ///double coupled_scalar = 0.0;
    ///MPI_Allreduce(&local_micro_value, &coupled_scalar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    betweenTime = std::chrono::steady_clock::now();
    

    {
      using LO = int;
      using GO = int;
      using map_type = Tpetra::Map<LO, GO>;
      using vec_type = Tpetra::Vector<double, LO, GO>;
      
      
      
using node_type = typename vec_type::node_type;
using device_type = typename vec_type::device_type;
using execution_space = typename vec_type::execution_space;
using memory_space = typename device_type::memory_space;

if (rank == 0) {
  std::cout << "\n-- Tpetra type information --\n";
  std::cout << "vec_type::node_type        = " << typeid(node_type).name() << '\n';
  std::cout << "vec_type::device_type      = " << typeid(device_type).name() << '\n';
  std::cout << "vec_type::execution_space  = " << typeid(execution_space).name() << '\n';
  std::cout << "vec_type::memory_space     = " << typeid(memory_space).name() << '\n';
  std::cout << '\n';
}
      
      

      auto comm = Teuchos::rcp(
        new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(MPI_COMM_WORLD)));

      const LO local_num_rows = 5;
      const Tpetra::global_size_t global_num_rows =
        static_cast<Tpetra::global_size_t>(local_num_rows) * size;

      auto map = Teuchos::rcp(new map_type(global_num_rows, local_num_rows, 0, comm));
      vec_type x(map);
/*
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
      */
  }
  
  
  {
    ryml::ConstNodeRef inputRoot;
    {
      std::string inputFileName = "/home/oesterle/rd/testTrilinosKokkos_Base/testTrilinosKokkos/input/in1.yaml";
      std::ifstream fin(inputFileName);
      if (!fin) throw std::runtime_error("Cannot open input file: " + inputFileName);
      std::stringstream ss;
      ss << fin.rdbuf();
      std::string inString = ss.str();
      ryml::Tree tree = ryml::parse_in_arena(c4::to_csubstr(inString));
      inputRoot = tree["in1"];
      
      const bool bool1 =
            rget<bool>(inputRoot, "bool1");
            
      std::cout<<"RYML: bool1="<<bool1<<"\n";
    }
  }
  
  
}
  
  const double t =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();

  if (!rank) {
    std::cout << "Total wall time: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n";
    std::cout << "betweenTime: " << std::chrono::duration<double>(betweenTime - startTime).count() << " s\n";
  }

  return 0;
}
