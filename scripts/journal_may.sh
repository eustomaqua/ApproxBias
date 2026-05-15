#!/bin/sh
#SBATCH --job-name=MyJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00

#The partition is the queue you want to run on. standard is gpu and can be omitted.
#number of independent tasks we are going to start in this script
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#Skipping many options! see man sbatch
# From here on, we can start our program


module load singularity
cd /home/qgl539/GitH/AB_15May

EXP=mCV_cvg1c
PRE=min_max
for DAT in ricci german ppr ppvr adult
do
    singularity exec /home/qgl539/Singdocker/enfair.sif bash -c "
         source /opt/conda/etc/profile.d/conda.sh
	 conda activate py311
	 python hfm_nonbin_exec.py -cvg may12 -pre $PRE -exp $EXP -dat $DAT -nk 5 -rep
    "
done


# ssh hendrix
# srun -p gpu --pty --time=2:00:00 --gres gpu:0 bash
# module load singularity
# cd Singdocker
# singularity run enfair.sif
# source activate py38|fmpar
# exit

# chmod +x ?.sh
# nohup ./?.sh
# ps aux | grep ApproxBias
# kill -9 *
# ps -ef | grep qgl539
