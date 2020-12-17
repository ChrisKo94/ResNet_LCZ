from __future__ import print_function
import argparse
import math
import matplotlib.pyplot as plt
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from model import LeNet_standard, LeNet_dropout
