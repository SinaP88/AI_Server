#!/bin/bash

cd /root/ai_server/3DDFA_Flask
export miniconda_dir=/root/miniconda
export FLASK_ENV=development
eval "$($miniconda_dir/bin/conda shell.bash hook)"
. $miniconda_dir/bin/activate 3DDFA
cd /root/ai_server/3DDFA_Flask/model
sh ./build.sh
cd /root/ai_server/3DDFA_Flask/
python app.py

