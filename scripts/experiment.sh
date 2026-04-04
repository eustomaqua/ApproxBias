#!/bin/sh
# source activate py38|fmpar


EXP=mCV_rexp4*
PRE=min_max
for DAT in ricci german ppr ppvr adult
do
    python hfm_nonbin_exec.py -rev -exp $EXP -pre $PRE -dat $DAT -nk 5 --nb-cls 7  -rep
    # python hfm_nonbin_exec.py -rev -exp $EXP -pre $PRE -dat $DAT -nk 5 --nb-cls 7  # cde
    
    python hfm_bin_exec.py -v ver4 -exp $EXP -pre $PRE -dat $DAT -nk 5 --nb-cls 7 -m1 25 -m2 11  # -rep
done


# ssh hendrix
# srun -p gpu --pty --time=2:00:00 --gres gpu:0 bash
# module load singularity
# cd Singdocker
# singularity run enfair.sif
# exit

# chmod +x ?.sh
# nohup ./?.sh
# ps aux | grep ApproxBias
# kill -9 *
# ps -ef | grep qgl539
