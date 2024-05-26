#!/bin/sh

deepspeed --hostfile=hostfile main_deepspeed.py --cuda --epochs 6 --model Transformer --lr 5 --log-interval 1 \
    --emsize 100 --nhead 2 --nhid 100 \
    --deepspeed --deepspeed_config ds_config.json
