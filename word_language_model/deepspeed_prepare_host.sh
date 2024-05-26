#!/bin/bash

#sudo ufw allow 29500
#sudo ufw enable
sudo ufw disable # NCCL uses wide range of ports like 30000 - 70000

sudo apt install ninja-build # So that ninja is in system folders, not in /home/<user>/.local...
