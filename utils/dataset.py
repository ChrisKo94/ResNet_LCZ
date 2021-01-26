import torch

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, image_data, labels):
        'Initialization'
        self.labels = labels
        self.image_data = image_data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_data)

  def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.image_data[index]
        y = self.labels[index]

        return X, y