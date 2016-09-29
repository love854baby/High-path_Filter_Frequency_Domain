#!/bin/sh

# Allocate 48 cores wherever they're available
#SBATCH -n 1

# Only use nodes with no other jobs on them
# This will minimize the number of different nodes
# used, and hence minimize message passing over the Infiniband network.
# It may delay job startup if the cluster is very busy, though, since it
# must wait for 4 completely empty nodes.
##SBATCH --exclusive

# Memory requirements: Default maximum is 256 megabytes per process.
# Specify more here if you need it.  Requirements should be overstimated
# SLIGHTLY.  Demanding too little memory will cause jobs to be terminated
# by the scheduler.  Demanding too much will prevent other jobs from using
# available resources and is inconsiderate to other users.  Specify either
# --mem or --mem-per-cpu, not both.

# Memory requirement per node
##SBATCH --mem=23000

# Memory requirement per process
##SBATCH --mem-per-cpu=1000

# The module command is defined in the shell startup scripts.
# Bourne-family shell scripts don't source startup scripts, so we have to
# do it manually.
. env.sh

mpirun -n 4 DFT florida_satellite.pgm result.pgm

