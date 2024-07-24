#!/bin/bash
p1=$1
p2=$2
seed=$3
cd ~/tutorial-pic
. ~/qvenv/bin/activate
echo $p1 $p2 $seed
python3 main.py --p1 $p1 --p2 $p2 --seed $seed
echo finished
deactivate