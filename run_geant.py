#!/usr/bin/env bash

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    return opt

args = parse_args()

os.system(f"python main.py \
--batch_size 64 \
--concur_size 96 \
--epoch_1 50000 \
--epoch_2 300 \
--loss_func_1 l1 \
--loss_func_2 l1 \
--dataset geant \
--dim_mults 1 2 4 \
--lr_1 1e-4 \
--hd 32 \
--init_num 500 \
--pre_ep 5000 \
--plot True \
--visualize tsne \
--nodes_num 23 \
--emb_size 12")