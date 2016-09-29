#!/bin/sh

# Exit script on first error
set -e

# Batch serial job
#SBATCH --ntasks=1

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

# Set up environment (PATH, LD_LIBRARY_PATH, etc.) for fluent
. env.sh

mpicxx -o DFT -O3 2DDFT.c