#!/bin/bash
set -e
export miniconda_dir=/root/miniconda
echo "========== Install Requirements ============"
# navigate to app directory
cd /root/ai_server/OPG_Flask
# activate the base
eval "$($miniconda_dir/bin/conda shell.bash hook)"
# create new conda environment OPG and all dependencies
conda env create -f environment.yml
# activate conda env OPG
conda activate OPG
# Doppel check if OPG acivated
. $miniconda_dir/bin/activate OPG
#echo "DBG: SHLVL=${SHLVL}"
echo "DBG: Value of CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"
# Installing MaskRCNN package using pip
pip install mrcnn
pip install waitress
echo "=========== Let's run the App =============="
#python app.py
