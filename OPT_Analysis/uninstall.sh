#!/bin/bash

export miniconda_dir=/root/miniconda
systemctl stop opgflask
cd /root
. $miniconda_dir/bin/deactivate OPG
yes | conda remove --name OPG --all


