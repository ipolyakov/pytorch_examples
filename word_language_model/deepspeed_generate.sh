#!/bin/bash

python3 generate.py --load_dir . --ckpt_id model.pt --deepspeed --deepspeed_config ds_config.json --cuda
