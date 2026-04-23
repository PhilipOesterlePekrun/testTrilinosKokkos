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

#include <mirco_evaluate.h>
#include <mirco_inputparameters.h>
#include <mirco_kokkostypes.h>
#include <mirco_topologyutilities.h>

#ifdef use_ryml
#include <ryml.hpp>
#include <ryml_std.hpp>
template<class T>
T rget(ryml::ConstNodeRef node, c4::csubstr key)
{
  auto child = node.find_child(key);
  if (!child.readable())
    throw std::runtime_error(
        "Parameter \"" + std::string(key.str, key.len) + "\" not found");

  T value{};
  if (!ryml::read(child, &value))
    throw std::runtime_error(
        "Parameter \"" + std::string(key.str, key.len) + "\" has invalid value");

  return value;
}
#endif

int MPI_TpetraSerial_KokkosCuda(int argc, char* argv[])
{
  const auto startTime = std::chrono::steady_clock::now();
  
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Pure Kokkos
  {
    using namespace MIRCO;
    
    if(!rank) {
      std::cout << "-- Kokkos information --\n";
      std::cout << "Threads in use: " << ExecSpace_Default_t().concurrency() << "\n";
      std::cout << "Default execution space: " << typeid(ExecSpace_Default_t).name() << "\n";
      std::cout << "Default host execution space: " << typeid(ExecSpace_DefaultHost_t).name() << "\n";
      std::cout << "Default memory space: " << typeid(MemorySpace_ofDefaultExec_t).name() << "\n";
      std::cout << "Default host memory space: " << typeid(MemorySpace_Host_t).name() << "\n";
      std::cout << "\n";
      
      std::cout << "num devices = " << Kokkos::num_devices() << '\n';
    }

    InputParameters inputParams(1, 1, 0.3, 0.3, 1e-6, 10/*delta*/, 1000/*laterallength*/, 7/*resolution*/, 10/*stddev*/, 0.5/*hurst*/, 100, false, true, false, 120);

    ViewVector_d meshgrid = CreateMeshgrid(inputParams.N, inputParams.grid_size);
    const double topologyMax = GetMax(inputParams.topology);

    // Main evaluation agorithm; MIRCO uses Kokkos::parallel_for() and Kokkos-Kernels
    double meanPressure, effectiveContactAreaFraction;
    if(!rank) Evaluate(meanPressure, effectiveContactAreaFraction, inputParams, topologyMax, meshgrid);
  }

  const auto betweenTime = std::chrono::steady_clock::now();
  
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

    const LO local_num_rows = 5;
    const Tpetra::global_size_t global_num_rows =
      static_cast<Tpetra::global_size_t>(local_num_rows) * size;

    auto map = Teuchos::rcp(new map_type(global_num_rows, local_num_rows, 0, comm));
    vec_type x(map);
  }
  
  #ifdef use_ryml
  {
    ryml::ConstNodeRef inputRoot;
    {
      std::string inputFileName = "../testTrilinosKokkos/input/in1.yaml";
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
  #endif
  
  const double t =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();

  if (!rank) {
    std::cout << "MPI_TpetraSerial_KokkosCuda betweenTime: " << std::chrono::duration<double>(betweenTime - startTime).count() << " s\n";
    std::cout << "MPI_TpetraSerial_KokkosCuda total wall time: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n";
  }

  return 0;
}
