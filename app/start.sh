#!/bin/bash

cd /root/ai_server/OPG_Flask
export miniconda_dir=/root/miniconda
eval "$($miniconda_dir/bin/conda shell.bash hook)"
. $miniconda_dir/bin/activate OPG
python app.py

