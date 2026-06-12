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

#include <Amesos.h>
#include <Amesos_BaseSolver.h>

#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Map.h>
#include <Epetra_SerialComm.h>
#include <Epetra_Vector.h>

#include <KokkosLapack_gesv.hpp>

#include <Teuchos_TimeMonitor.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <typeinfo>






#define TEST_HAVE_OPENBLAS_THREAD_CONTROL 1


#ifdef TEST_HAVE_OPENBLAS_THREAD_CONTROL
extern "C" {
  char* openblas_get_config();
  int openblas_get_num_threads();
  void openblas_set_num_threads(int);
}
#endif

class ScopedBlasThreads
{
 public:
  explicit ScopedBlasThreads(const int num_threads)
#ifdef TEST_HAVE_OPENBLAS_THREAD_CONTROL
    : old_num_threads_(openblas_get_num_threads())
#endif
  {
#ifdef TEST_HAVE_OPENBLAS_THREAD_CONTROL
    openblas_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif
  }

  ~ScopedBlasThreads()
  {
#ifdef TEST_HAVE_OPENBLAS_THREAD_CONTROL
    openblas_set_num_threads(old_num_threads_);
#endif
  }

  static void set_num_threads(const int num_threads)
  {
#ifdef TEST_HAVE_OPENBLAS_THREAD_CONTROL
    openblas_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif
  }

  static std::string thread_info()
  {
#ifdef TEST_HAVE_OPENBLAS_THREAD_CONTROL
    return std::to_string(openblas_get_num_threads());
#else
    return "OpenBLAS thread control unavailable";
#endif
  }

  static void print_info()
  {
#ifdef TEST_HAVE_OPENBLAS_THREAD_CONTROL
    std::cout << "OpenBLAS config: " << openblas_get_config() << "\n";
    std::cout << "OpenBLAS threads: " << openblas_get_num_threads() << "\n";
#else
    std::cout << "OpenBLAS thread control unavailable\n";
#endif
  }

 private:
#ifdef TEST_HAVE_OPENBLAS_THREAD_CONTROL
  int old_num_threads_;
#endif
};

void print_proc_status(const std::string& label, const int world_rank)
{
  if (world_rank != 0) return;

  std::ifstream status("/proc/self/status");
  std::string line;

  std::cout << "-- " << label << " --\n";

  while (std::getline(status, line)) {
    if (line.rfind("Threads:", 0) == 0 ||
        line.rfind("Cpus_allowed_list:", 0) == 0 ||
        line.rfind("voluntary_ctxt_switches:", 0) == 0 ||
        line.rfind("nonvoluntary_ctxt_switches:", 0) == 0) {
      std::cout << line << '\n';
    }
  }

  std::cout << '\n';
}

#include <dirent.h>
#include <sched.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <vector>

cpu_set_t make_full_node_cpu_set()
{
  cpu_set_t mask;
  CPU_ZERO(&mask);

  const long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
  for (int cpu = 0; cpu < static_cast<int>(num_cpus); ++cpu) {
    CPU_SET(cpu, &mask);
  }

  return mask;
}

std::vector<pid_t> get_process_thread_ids()
{
  std::vector<pid_t> tids;

  DIR* dir = opendir("/proc/self/task");
  if (dir == nullptr) {
    return tids;
  }

  while (dirent* entry = readdir(dir)) {
    if (entry->d_name[0] == '.') {
      continue;
    }

    tids.push_back(static_cast<pid_t>(std::stoi(entry->d_name)));
  }

  closedir(dir);
  return tids;
}



#include <cstdlib>
#include <string>

#include <cstdlib>
#include <string>
#include <unistd.h>

void set_all_thread_affinity(const std::string& cpu_list)
{
  const std::string pid = std::to_string(getpid());

  const std::string cmd =
      "for t in /proc/" + pid + "/task/*; do "
      "taskset -pc " + cpu_list + " ${t##*/} >/dev/null; "
      "done";

  std::system(cmd.c_str());
}

#include <thread>
void sleepy_barrier(MPI_Comm comm)
{
  MPI_Request request;
  MPI_Ibarrier(comm, &request);

  int done = 0;
  while (!done) {
    MPI_Test(&request, &done, MPI_STATUS_IGNORE);

    if (!done) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

class ScopedAllThreadAffinity
{
 public:
  explicit ScopedAllThreadAffinity(const cpu_set_t& new_mask)
  {
    const auto tids = get_process_thread_ids();

    for (const pid_t tid : tids) {
      cpu_set_t old_mask;
      CPU_ZERO(&old_mask);

      if (sched_getaffinity(tid, sizeof(cpu_set_t), &old_mask) == 0) {
        old_masks_.push_back({tid, old_mask});
      }

      if (sched_setaffinity(tid, sizeof(cpu_set_t), &new_mask) != 0) {
        std::cerr << "sched_setaffinity failed for tid " << tid
                  << ": " << std::strerror(errno) << '\n';
      }
    }
  }

  ~ScopedAllThreadAffinity()
  {
    for (const auto& [tid, old_mask] : old_masks_) {
      sched_setaffinity(tid, sizeof(cpu_set_t), &old_mask);
    }
  }

 private:
  std::vector<std::pair<pid_t, cpu_set_t>> old_masks_;
};


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
  
  
  
  
  
  if(!world_rank) {
    std::cout << "-- MPI information --\n";
    std::cout << "world_size: " << world_size << "\n\n";
  }

  Kokkos::ScopeGuard kokkos_guard(argc, argv);
  
  
  
  ScopedBlasThreads::set_num_threads(1);
  if (!world_rank) {
    std::cout << "-- BLAS thread control information --\n";
    ScopedBlasThreads::print_info();
    std::cout << "\n";
  }
  
    set_all_thread_affinity(std::to_string(node_rank));
  
  
  // Tpetra (with host NO because Tpetra_INST_CUDA=OFF)
  {
    set_all_thread_affinity(std::to_string(node_rank));
    const auto startTime = std::chrono::steady_clock::now();
    
    using LO = int;
    using GO = int;
    
    /*
    using map_type = Tpetra::Map<LO, GO>;
    using vec_type = Tpetra::Vector<double, LO, GO>;
    */
    using solver_node_type =
    Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::Serial>;
    using map_type = Tpetra::Map<LO, GO, solver_node_type>;
    using vec_type = Tpetra::Vector<double, LO, GO, solver_node_type>;
    
    
    
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
      std::cout << "Tpetra section total time:" << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n";
    }
    
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  
  
  
  
  if(!world_rank)
    std::cout<<"\n\n\n\n\n";
  
  
  
  print_proc_status("before UMFPACK", world_rank);
  // Amesos / UMFPACK test, with OpenBLAS forced to serial
{
  ScopedBlasThreads blas_threads(1);
  set_all_thread_affinity(std::to_string(node_rank));

  const auto startTime = std::chrono::steady_clock::now();

  Epetra_SerialComm serial_comm;
  const int n = 20000;

  Epetra_Map map(n, 0, serial_comm);
  Epetra_CrsMatrix A(Copy, map, 3);

  for (int row = 0; row < n; ++row) {
    int cols[3];
    double vals[3];
    int num_entries = 0;

    if (row > 0) {
      cols[num_entries] = row - 1;
      vals[num_entries] = -1.0;
      ++num_entries;
    }

    cols[num_entries] = row;
    vals[num_entries] = 2.0;
    ++num_entries;

    if (row + 1 < n) {
      cols[num_entries] = row + 1;
      vals[num_entries] = -1.0;
      ++num_entries;
    }

    A.InsertGlobalValues(row, num_entries, vals, cols);
  }

  A.FillComplete();

  Epetra_Vector x(map);
  Epetra_Vector b(map);

  x.PutScalar(0.0);
  b.PutScalar(1.0);

  Epetra_LinearProblem problem(&A, &x, &b);

  Amesos factory;
  const char* solver_name = "Amesos_Umfpack";

  if (!factory.Query(solver_name)) {
    if (!world_rank) {
      std::cout << "UMFPACK test skipped: " << solver_name << " not available\n\n";
    }
  } else {
    std::unique_ptr<Amesos_BaseSolver> solver(factory.Create(solver_name, problem));

    const int symbolic_status = solver->SymbolicFactorization();
    const int numeric_status = solver->NumericFactorization();
    const int solve_status = solver->Solve();

    double x_norm = 0.0;
    x.Norm2(&x_norm);

    if (!world_rank) {
      std::cout << "-- UMFPACK test --\n";
      std::cout << "OpenBLAS threads during UMFPACK: " << ScopedBlasThreads::thread_info() << '\n';
      std::cout << "SymbolicFactorization status: " << symbolic_status << '\n';
      std::cout << "NumericFactorization status: " << numeric_status << '\n';
      std::cout << "Solve status: " << solve_status << '\n';
      std::cout << "x.Norm2() = " << x_norm << '\n';
      std::cout << "UMFPACK time: "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count()
                << " s\n\n";
    }
  }
}
MPI_Barrier(MPI_COMM_WORLD);
  print_proc_status("after UMFPACK", world_rank);
  
  
  
  
  
  
  
  
  
  if(!world_rank)
    std::cout<<"\n\n\n\n\n";
  
  
    Kokkos::fence();
set_all_thread_affinity("0-63");
Kokkos::fence();
  // Pure Kokkos
  {
    const auto startTime = std::chrono::steady_clock::now();
    
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
          
        Kokkos::fence();

const cpu_set_t full_node_mask = make_full_node_cpu_set();
ScopedAllThreadAffinity full_node_affinity(full_node_mask);
          
        
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
          << "checksum="<<std::sqrt(checksum)
          << ", time=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count()
                    << " s\n";
          
          
          
        
          
          
        // KokkosKernels dense LAPACK test, with OpenBLAS temporarily threaded
        {
          ScopedBlasThreads blas_threads(omp_get_max_threads());
          
          std::cout<<"(omp_get_max_threads() = "<<omp_get_max_threads()<<" = blas_threads)\n\n";

          using DenseMatrix_d = ViewMatrix_d;
          using DenseVector_i = Kokkos::View<int*, Kokkos::LayoutLeft, Device_Default_t>;

          const int dense_n = 20000;
          const int nrhs = 1;

          DenseMatrix_d A("A_dense", dense_n, dense_n);
          DenseMatrix_d B("B_dense", dense_n, nrhs);
          DenseVector_i piv("piv", dense_n);

          Kokkos::parallel_for(
              "init_dense_system",
              Kokkos::MDRangePolicy<ExecSpace_Default_t, Kokkos::Rank<2>>({0, 0}, {dense_n, dense_n}),
              KOKKOS_LAMBDA(const int i, const int j) {
                if (i == j) {
                  A(i, j) = 4.0;
                } else {
                  A(i, j) = 1.0 / static_cast<double>(dense_n + 1 + Kokkos::abs(i - j));
                }
              });

          Kokkos::parallel_for(
              "init_dense_rhs",
              Kokkos::RangePolicy<ExecSpace_Default_t>(0, dense_n),
              KOKKOS_LAMBDA(const int i) {
                B(i, 0) = 1.0 + 0.001 * static_cast<double>(i % 17);
              });

          Kokkos::fence();

          const auto startTimeGesv = std::chrono::steady_clock::now();

          KokkosLapack::gesv(A, B, piv);

          Kokkos::fence();

          double solution_norm2 = 0.0;
          Kokkos::parallel_reduce(
              "dense_solution_norm",
              Kokkos::RangePolicy<ExecSpace_Default_t>(0, dense_n),
              KOKKOS_LAMBDA(const int i, double& local_sum) {
                local_sum += B(i, 0) * B(i, 0);
              },
              solution_norm2);

          std::cout << "KokkosKernels gesv work (node_rank=" << node_rank
                    << "; world_rank=" << world_rank << "): "
                    << "BLAS threads=" << ScopedBlasThreads::thread_info()
                    << ", ||x||=" << std::sqrt(solution_norm2)
                    << ", time=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTimeGesv).count()
                    << " s\n\n";
        }
          
          
          
          
          
        std::cout << "\tTOTAL KOKKOS TIME: "<<std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n\n";
      }
      
      sleepy_barrier(comm_node);//MPI_Barrier(comm_node);
    }
    Kokkos::fence();
set_all_thread_affinity(std::to_string(node_rank));
Kokkos::fence();
    if (!world_rank) {
      std::cout << "Kokkos section total time:" << std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() << " s\n";
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}
