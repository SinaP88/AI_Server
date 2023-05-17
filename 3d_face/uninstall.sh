#!/bin/bash

export miniconda_dir=/root/miniconda
systemctl stop face3dflask
cd /root
. $miniconda_dir/bin/deactivate 3DDFA
yes | conda remove --name 3DDFA --all


