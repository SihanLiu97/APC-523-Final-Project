#!/bin/bash
#SBATCH --job-name=fp            # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=10G                # memory per cpu-core (4G is default)
#SBATCH --time=1-00:00:00        # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=sl7434@princeton.edu

module load anaconda3/2024.2

python 2D_RK4.py   5   5   5   5   5   5
python 2D_RK4.py   4   4   6   4   9   4
python 2D_RK4.py   4   4   4   4   8.2 8.2
python 2D_RK4.py   7.5 7.5 3.5 7.5 3.5 3.5
python 2D_RK4.py   3   6.5 4   3   6.5 3
python 2D_RK4.py   2.5 5.5 8.5 2.5 6.5 2.5
python 2D_RK4.py   2.2 4.6 7   2.2 9.4 2.2

python 2D_FFT.py   5   5   5   5   5   5
python 2D_FFT.py   4   4   6   4   9   4
python 2D_FFT.py   4   4   4   4   8.2 8.2
python 2D_FFT.py   7.5 7.5 3.5 7.5 3.5 3.5
python 2D_FFT.py   3   6.5 4   3   6.5 3
python 2D_FFT.py   2.5 5.5 8.5 2.5 6.5 2.5
python 2D_FFT.py   2.2 4.6 7   2.2 9.4 2.2
