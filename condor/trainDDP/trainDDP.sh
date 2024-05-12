#!/lusr/bin/bash
source ~/.bashrc
cd /u/marco/github/neural-cellular-automata
source venv/bin/activate
cd /u/marco/github/neural-cellular-automata/_9_post_thesis_optimizations
TORCH_USE_CUDA_DSA=1 python3 train.py