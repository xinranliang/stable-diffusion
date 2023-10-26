import numpy as np
from glob import glob
from PIL import Image
import argparse
import os 

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import torchmetrics
from torchmetrics.multimodal.clip_score import CLIPScore

from utils import SimpleDataset