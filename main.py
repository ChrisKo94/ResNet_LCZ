import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import random
import math
from operator import add
import time
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR

from models.resnet import resnet50
from utils.dataset import Dataset

path_data = "/data/lcz42_votes/"

train_data = h5py.File(path_data + "train_data.h5",'r')
x_train = np.array(train_data.get("x"))
# Switch dimensions of tensor to suit Pytorch
x_train = np.rollaxis(x_train, 3, 1)
x_train = torch.from_numpy(x_train)
y_train = np.array(train_data.get("y"))
y_train = torch.from_numpy(y_train)

test_data = h5py.File(path_data + "test_data.h5",'r')
x_test = np.array(test_data.get("x"))
# Switch dimensions of tensor to suit Pytorch
x_test = np.rollaxis(x_test, 3, 1)
x_test = torch.from_numpy(x_test)
y_test = np.array(test_data.get("y"))
y_test = torch.from_numpy(y_test)

n_input_channel = 10
n_class = 17

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50(n_input_channel, n_class).to(device)
model = model.cuda()

# Training settings
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Randomly sample testing data
idx = torch.randperm(x_test.shape[0])
x_test = x_test[idx].view(x_test.size())
y_test = y_test[idx].view(y_test.size())

train_loader = torch.utils.data.DataLoader(Dataset(x_train, y_train), batch_size = 256, shuffle=True)
test_loader = torch.utils.data.DataLoader(Dataset(x_test, y_test), batch_size = 256, shuffle=False)

epochs = 30
learning_rate = 0.001

optimizer = optim.Adam(params = model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

scheduler = StepLR(optimizer, step_size=2, gamma=0.75)

start_time = time.time()

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Run the training batches
    for b, (X_train, Y_train) in enumerate(train_loader):
        b += 1
        # Learning rate decay
        scheduler.step()

        X_train, Y_train = X_train.to(device, dtype=torch.float), Y_train.to(device)

        # Apply the model
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, torch.max(Y_train, 1)[1])

        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == torch.max(Y_train, 1)[1]).sum()
        trn_corr += batch_corr

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        if b % 120 == 0:
            print(
                f'epoch: {i + 1:2}  batch: {b:4} [{256 * b:6}/{len(x_train):6}]  loss: {loss.item():10.8f} accuracy: {trn_corr.item() * 100 / (256 * b):7.3f}%')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, Y_test) in enumerate(test_loader):
            X_test, Y_test = X_test.to(device, dtype=torch.float), Y_test.to(device)

            # Apply the model
            y_val = model.forward(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == torch.max(Y_test, 1)[1]).sum()

    loss = criterion(y_val, torch.max(Y_test, 1)[1])
    test_losses.append(loss)
    test_correct.append(tst_corr)

    print(
        f'epoch: {i + 1:2} testing loss: {loss.item():10.8f} testing accuracy: {tst_corr.item() * 100 / len(x_test):7.3f}%')

print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed