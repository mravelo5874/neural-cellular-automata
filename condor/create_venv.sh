#!/lusr/bin/bash
source ~/.bashrc
cd /u/marco/github/neural-cellular-automata
python3 -m venv venv
source venv/bin/activate
pip install --no-cache-dir numpy 
pip install --no-cache-dir torch torchvision torchaudio
pip install --no-cache-dir matplotlib
pip install --no-cache-dir requests
pip install --no-cache-dir moviepy
pip install --no-cache-dir scipy
pip install --no-cache-dir gitpython
pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable"