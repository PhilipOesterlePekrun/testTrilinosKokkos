#!/bin/bash -l
#SBATCH --job-name=openmp_oversub_2node
#SBATCH --exclusive
#SBATCH --constraint=h100
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=out
#SBATCH --error=err

cd /home/oesterle/rd/testTrilinosKokkos_Base/build_openmp

module load cmake
module load openmpi/4.1.8
module load suite-sparse/5.4.0

export MY_TRILINOS_INSTALL=/home/oesterle/rd/Trilinos_Base/release_Upstream_E_int_full_kserial+openmp_openblasOpenMP/install
export LD_LIBRARY_PATH=${MY_TRILINOS_INSTALL}/lib:${LD_LIBRARY_PATH}

export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export KOKKOS_DISABLE_WARNINGS=1

export OMPI_MCA_mpi_yield_when_idle=1

scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $1 " slots=64"}' > hosts.2nodes
tac hosts.2nodes > hosts.2nodes.swapped

MPI_COMMON="
  -np 128
  --bind-to none
  --map-by ppr:64:node:OVERSUBSCRIBE
  --mca mpi_yield_when_idle 1
  -x OMP_NUM_THREADS
  -x OMP_PROC_BIND
  -x KOKKOS_DISABLE_WARNINGS
  -x OMPI_MCA_mpi_yield_when_idle
  -x LD_LIBRARY_PATH
"

echo "Hosts:"
cat hosts.2nodes
echo

echo "Placement/env diagnostic:"
mpirun \
  $MPI_COMMON \
  --hostfile hosts.2nodes \
  bash -lc 'echo host=$(hostname) rank=$OMPI_COMM_WORLD_RANK local=$OMPI_COMM_WORLD_LOCAL_RANK cpus=$(grep Cpus_allowed_list /proc/self/status | cut -f2) OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES' \
  | sort

echo
echo "Running executable, normal host order:"
mpirun \
  $MPI_COMMON \
  --hostfile hosts.2nodes \
  /home/oesterle/rd/testTrilinosKokkos_Base/build_openmp/exe_SERIALIZE

echo
echo "Running executable, swapped host order:"
mpirun \
  $MPI_COMMON \
  --hostfile hosts.2nodes.swapped \
  /home/oesterle/rd/testTrilinosKokkos_Base/build_openmp/exe_SERIALIZE
  