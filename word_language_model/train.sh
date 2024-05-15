#!/bin/sh

python3 main_deepspeed.py --cuda --epochs 6 --model Transformer --lr 5 --log-interval 1
#python3 main.py --cuda --epochs 6 --model Transformer --lr 5 --log-interval 1
