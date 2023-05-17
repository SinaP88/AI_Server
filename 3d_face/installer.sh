#!/bin/bash
set -e
miniconda_path=~/miniconda
echo "========== Install Requirements ============"
eval "$($miniconda_path/bin/conda shell.bash hook)"
cd /root/ai_server/3DDFA_Flask/
conda create --name 3DDFA python=3.6 -y

. $miniconda_path/etc/profile.d/conda.sh
. $miniconda_path/bin/activate 3DDFA

echo "DBG: SHLVL=${SHLVL}"
echo "DBG: Value of CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"

cd /root/ai_server/3DDFA_Flask/model
pip install -r requirements.txt
sh ./build.sh
echo "=========== Requirements successfully installed! =============="


