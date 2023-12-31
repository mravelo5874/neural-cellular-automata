#!/lusr/bin/bash
source ~/.bashrc
cd /u/marco/github/neural-cellular-automata
python3 -m venv venv
source venv/bin/activate
pip install numpy
pip install torch
pip install matplotlib
pip install requests
pip install moviepy
pip install scipy
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"