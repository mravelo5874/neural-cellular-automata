#!/lusr/bin/bash
source ~/.bashrc
# eval "$('/u/jmz679/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# conda activate /scratch/cluster/jmz679/envs/automatic
# accelerate launch videonet_training.py
cd /u/marco/github/neural-cellular-automata/5_voxel_nca
python3 train.py