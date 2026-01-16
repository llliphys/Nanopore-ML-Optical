#!/bin/bash

#SBATCH --job-name=OPTIC
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -o ./%J.out
#SBATCH -e ./%J.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module reset
module load phys/Siesta/5.2.2-foss-2023b-mpi

# rm -rf WFS.nc
# ln -s /dev/shm/WFS.nc WFS.nc
     
srun siesta < optic.fdf > output
