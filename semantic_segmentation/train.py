import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.tensorboard
from dataloader import *
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser(description="semantic segmentation for AugusTOOTH")

parser.add_argument("--mode", default="train", choices=["train","validation", "test"], type=str, dest="mode")
parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--ny", default=256, type=int, dest="ny")
parser.add_argument("--nx", default=256, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--network", default="FCN", choices=["FCN"], type=str, dest="network")

args = parser.parse_args()

mode = args.mode
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker
network = args.network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')