#!/lusr/bin/bash
source ~/.bashrc
cd /u/marco/github/neural-cellular-automata
python3 -m venv venv
source venv/bin/activate
pip install --no-cache-dir -r condor/requirements.txt