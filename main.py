import numpy as np
import h5py
import time
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from models.resnet import resnet50
from utils.dataset import Dataset
from utils.early_stopping import EarlyStopping

path_data = "/data/lcz42_votes/data/"
#path_data = "E:/Dateien/LCZ Votes/"

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

# set parameters
n_epochs = 100
learning_rate = 0.001
patience = 10
batch_size = 256

train_loader = torch.utils.data.DataLoader(Dataset(x_train, y_train), batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(Dataset(x_test, y_test), batch_size = batch_size, shuffle=False)


optimizer = optim.Adam(params = model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=2, gamma=0.75)

def train_model(model, batch_size, patience, n_epochs):

    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for i in range(n_epochs):
        trn_corr = 0
        tst_corr = 0

        model.train()
        # Run the training batches
        for b, (X_train, Y_train) in enumerate(train_loader):
            b += 1

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

            # Learning rate decay
            scheduler.step()

            train_losses.append(loss.item())
            train_correct.append(trn_corr)

        model.eval()
        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, Y_test) in enumerate(test_loader):
                X_test, Y_test = X_test.to(device, dtype=torch.float), Y_test.to(device)

                # Apply the model
                y_val = model.forward(X_test)
                loss = criterion(y_val, torch.max(Y_test, 1)[1])

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1]
                batch_corr = (predicted == torch.max(Y_test, 1)[1]).sum()
                tst_corr += batch_corr


                test_losses.append(loss.item())
                test_correct.append(tst_corr)

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(test_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print(
            f'epoch: {i + 1:2} training loss: {train_loss:10.8f} training accuracy: {trn_corr.item() * 100 / len(x_train) :7.3f}%')
        print(
            f'epoch: {i + 1:2} validation loss: {valid_loss:10.8f} validation accuracy: {tst_corr.item() * 100 / len(x_test):7.3f}%')

        # clear lists to track next epoch
        train_losses = []
        test_losses = []

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses

start_time = time.time()
model, train_loss, valid_loss = train_model(model, batch_size, patience, n_epochs)
print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed

PATH = "ResNet50_b" + batch_size + "_e_" + n_epochs + "_weightdecay"

torch.save(model.state_dict(), PATH)