#!/bin/bash
#SBATCH -n 1
#SBATCH -o slurm-output/winograd-job-%j.out
#SBATCH -e slurm-error/winograd-job-%j.err
#SBATCH -c 64
#SBATCH --exclusive
#SBATCH --exclude hepnode0
#SBATCH --gres=gpu:L40:2
# Note: How to run this script on slurm: `sbatch ./run.sh'.
# Note: see `man sbatch' for more options.

# Note: Manual control number of omp threads
# export OMP_NUN_THREADS=64

# Note: numactl - Control NUMA policy for processes or shared memory, see `man numactl'.`
# Note: perf-stat - Run a command and gather performance counter statistics, see `man perf stat'.
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && numactl --cpunodebind=0-3 --membind=0-3 perf stat -ddd nsys profile --stats=true -o winograd1 ./winograd conf/vgg16.conf