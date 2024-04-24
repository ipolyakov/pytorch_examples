#!/bin/sh

deepspeed main_deepspeed.py --cuda --epochs 6 --model Transformer --lr 5 --deepspeed --deepspeed_config ds_config.json
