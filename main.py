import numpy as np
import h5py
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

from models.resnet import resnet18
from utils.dataset import Dataset
from utils.early_stopping import EarlyStopping
from utils.avg_accuracy import get_avg_accuracy

from sklearn.metrics import cohen_kappa_score
import statistics

if torch.cuda.is_available():
    path_data = "/data/lcz42_votes/data/"
else:
    path_data = "D:/Data/LCZ_Votes/"
# path_data = "E:/Dateien/LCZ Votes/"


#mode = "all"
mode = "urban"
weights = False
lr_decay = "cycle"
#lr_decay = "step"

entropy_quantile = 0.2 # choose quantile of most certain images (w.r.t. voter entropy) for training, requires mode = "urban"

train_data = h5py.File(path_data + "train_data.h5", 'r')
x_train = np.array(train_data.get("x"))
# Switch dimensions of tensor to suit Pytorch
x_train = np.rollaxis(x_train, 3, 1)
x_train = torch.from_numpy(x_train)
y_train = np.array(train_data.get("y"))
y_train = torch.from_numpy(y_train)

test_data = h5py.File(path_data + "test_data.h5", 'r')
x_test = np.array(test_data.get("x"))
# Switch dimensions of tensor to suit Pytorch
x_test = np.rollaxis(x_test, 3, 1)
x_test = torch.from_numpy(x_test)
y_test = np.array(test_data.get("y"))
y_test = torch.from_numpy(y_test)

if mode == "urban":
    indices_train = np.where((torch.max(y_train, 1)[1] + 1).numpy() < 11)[0]
    indices_test = np.where((torch.max(y_test, 1)[1] + 1).numpy() < 11)[0]
    x_train = x_train[indices_train,:,:,:]
    y_train = y_train[indices_train]
    x_test = x_test[indices_test,:,:,:]
    y_test = y_test[indices_test]

if entropy_quantile > 0 and mode == "urban":
    indices_train = np.where((torch.max(y_train, 1)[1] + 1).numpy() < 11)[0]
    entropies = h5py.File(path_data + "entropies_train.h5", 'r')
    entropies_train = np.array(entropies.get("entropies_train"))
    entropies_train = entropies_train[indices_train]
    entropies_train[np.where(np.isnan(entropies_train))] = 0
    entropies = pd.DataFrame({"entropies": entropies_train,
                              "order": np.arange(len(y_train))})
    entropies = entropies.sort_values(by=['entropies'])
    ## Order training data accordingly
    idx = np.array(entropies["order"])
    ## Cut off at given quantile
    idx = idx[:np.floor(entropy_quantile * len(idx)).astype(int)]
    x_train = x_train[idx, :, :, :]
    y_train = y_train[idx]

n_input_channel = 10
if mode == "urban":
    n_class = 10
else:
    n_class = 17

if mode == "urban":
    class_weights = [47.18229167, 1.89618001, 8.41914498, 14.27738377, 1.70170001,
                     1., 6.49158008, 1.91846675, 15.32825719, 6.05953177]
else:
    class_weights = [193.22916667, 7.76556777, 34.4795539, 58.47123719, 6.96909928,
                     4.09537477, 26.58545324,  7.85684032, 62.7749577, 24.81605351,
                     3.27666151, 6.3020214, 62.19614417, 1.4974773, 52.69886364,
                     83.74717833, 1.]

if torch.cuda.is_available():
    class_weights = torch.FloatTensor(class_weights).cuda()
else:
    class_weights = torch.FloatTensor(class_weights)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet18(n_input_channel, n_class).to(device)

if torch.cuda.is_available():
    model = model.cuda()

# Testing shuffle settings
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Randomly sample testing data
idx = torch.randperm(x_test.shape[0])
x_test = x_test[idx].view(x_test.size())
y_test = y_test[idx].view(y_test.size())

# set parameters
n_epochs = 100
learning_rate = 0.000001
patience = 20
batch_size = 128

if lr_decay == "cycle":
    PATH = "/data/lcz42_votes/ResNet_LCZ/ResNet18_b" + str(batch_size) + "_e_" + str(n_epochs) + "_cyclicweightdecay"
else:
    PATH = "/data/lcz42_votes/ResNet_LCZ/ResNet18_b" + str(batch_size) + "_e_" + str(n_epochs) + "_stepweightdecay"

if mode == "urban":
    PATH = PATH + "_urban"

if entropy_quantile > 0:
    PATH = PATH + "_most_certain_" + str(entropy_quantile)

train_loader = torch.utils.data.DataLoader(Dataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(Dataset(x_test, y_test), batch_size=batch_size, shuffle=False)

label_table_train = (np.argmax(y_train, axis=1) + 1).numpy()
label_table_train = pd.DataFrame(np.transpose(np.unique(label_table_train, return_counts=True)),
                                 columns=["class", "sum"]).astype(float)
label_table_test = (np.argmax(y_test, axis=1) + 1).numpy()
label_table_test = pd.DataFrame(np.transpose(np.unique(label_table_test, return_counts=True)),
                                columns=["class", "sum"]).astype(float)
if mode == "urban":
    init_label_table = pd.DataFrame({"class": np.arange(1, 11), "correct_sum": np.zeros(10)})
else:
    init_label_table = pd.DataFrame({"class": np.arange(1, 18), "correct_sum": np.zeros(17)})

optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

if weights == True:
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()

if lr_decay == "cycle":
    scheduler = CyclicLR(optimizer, base_lr=learning_rate, max_lr=0.0001, mode='exp_range', gamma=0.9,
                         cycle_momentum=False)
else:
    scheduler =optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

def train_model(model, batch_size, patience, n_epochs):
    train_losses = []
    test_losses = []
    # train_correct = []
    # test_correct = []
    train_kappa = []
    test_kappa = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=PATH + '_checkpoint.pt')

    for i in range(n_epochs):
        torch.manual_seed(42 + 42 * i)
        torch.cuda.manual_seed(42 + 42 * i)
        trn_corr = 0
        tst_corr = 0
        running_label_table_train = init_label_table
        running_label_table_test = init_label_table
        train_acc_diff = []
        test_avg_accuracy = []

        model.train()
        # Run the training batches
        for b, (X_train, Y_train) in enumerate(train_loader):
            b += 1

            X_train, Y_train = X_train.to(device, dtype=torch.float), Y_train.to(device)

            # Apply the model
            y_pred = model.forward(X_train)
            loss = criterion(y_pred, torch.max(Y_train, 1)[1])

            # Tally the number of correct predictions
            actual = torch.max(Y_train, 1)[1] + 1
            predicted = torch.max(y_pred.data, 1)[1] + 1
            batch_corr = (predicted == actual).sum()
            trn_corr += batch_corr
            # compute kappa
            actual = actual.cpu().numpy()
            predicted = predicted.cpu().numpy()
            train_kappa.append(cohen_kappa_score(actual, predicted))
            # derive categorical accuracy
            running_label_table_train = get_avg_accuracy(actual, predicted, running_label_table_train)
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Learning rate decay
            scheduler.step()

            train_losses.append(loss.item())
            # train_correct.append(trn_corr)

        model.eval()

        torch.manual_seed(42 + 42 * i + 1)
        torch.cuda.manual_seed(42 + 42 * i + 1)

        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, Y_test) in enumerate(test_loader):
                X_test, Y_test = X_test.to(device, dtype=torch.float), Y_test.to(device)

                # Apply the model
                y_val = model.forward(X_test)
                loss = criterion(y_val, torch.max(Y_test, 1)[1])

                # Tally the number of correct predictions
                actual = torch.max(Y_test, 1)[1] + 1
                predicted = torch.max(y_val.data, 1)[1] + 1
                batch_corr = (predicted == actual).sum()
                tst_corr += batch_corr
                # Compute kappa
                actual = actual.cpu().numpy()
                predicted = predicted.cpu().numpy()
                test_kappa.append(cohen_kappa_score(actual, predicted))
                # derive categorical accuracy
                running_label_table_test = get_avg_accuracy(actual, predicted, running_label_table_test)

                test_losses.append(loss.item())
                # test_correct.append(tst_corr)

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(test_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            train_kappa = np.mean(train_kappa)
            test_kappa = np.mean(test_kappa)
            train_acc_diff = running_label_table_train["correct_sum"] / label_table_train["sum"]
            train_avg_accuracy = np.mean(train_acc_diff)
            test_acc_diff = running_label_table_test["correct_sum"] / label_table_test["sum"]
            test_avg_accuracy = np.mean(test_acc_diff)

        print(
            f'epoch: {i + 1:2} training loss: {train_loss:10.8f} training accuracy: {trn_corr.item() * 100 / len(x_train) :7.3f}%')
        print(
            f'training kappa: {train_kappa:7.3f} training average accuracy: {train_avg_accuracy * 100 :7.3f}%')
        print(
            f'epoch: {i + 1:2} validation loss: {valid_loss:10.8f} validation accuracy: {tst_corr.item() * 100 / len(x_test):7.3f}%')
        print(
            f'validation kappa: {test_kappa:7.3f} validation average accuracy: {test_avg_accuracy * 100 :7.3f}%')
        # clear lists to track next epoch
        train_losses = []
        test_losses = []
        train_kappa = []
        test_kappa = []
        train_acc_diff = []
        test_acc_diff = []

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(PATH + '_checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses


start_time = time.time()
model, train_loss, valid_loss = train_model(model, batch_size, patience, n_epochs)
print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed

torch.save(model.state_dict(), PATH)
np.save("/data/lcz42_votes/ResNet_LCZ/train_loss.npy", train_loss)
np.save("/data/lcz42_votes/ResNet_LCZ/test_loss.npy", valid_loss)
